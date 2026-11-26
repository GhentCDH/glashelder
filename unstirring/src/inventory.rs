use std::{
    collections::HashSet,
    fs::{File, OpenOptions},
    io::{BufRead as _, BufReader, Read as _, Seek as _, SeekFrom, Write as _},
    path::{Path, PathBuf},
};

use futures::{Stream, TryStreamExt as _, stream};
use reqwest::Url;
use serde::{Deserialize, Serialize};

use crate::{UnstirClient, image::print_image_bytes};

const BASE_INVENTORY: &str = "https://inventaris.onroerenderfgoed.be";
const BASE_IMAGES: &str = "https://beeldbank.onroerenderfgoed.be";
const TYPES: &str = "https://id.erfgoed.net/thesauri/erfgoedtypes/";
const PAGE: usize = 50;

/// `bouwkundig erfgoed`
const ROOT: &str = "147";

static FILTERS: [(&str, &str); 2] = [
    ("omvang", "1"),     // Element
    ("discipline", "2"), // Bouwkundig
];

#[derive(Deserialize, Serialize, Debug)]
pub struct ErfgoedObject {
    id: u32,
    uri: String,
    naam: String,
    primaire_foto: Option<String>,
    locatie_samenvatting: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Detail {
    kenmerkgroepen: Vec<Groep>,
}
#[derive(Deserialize, Debug)]
struct Groep {
    thesaurus: Vec<Term>,
}
#[derive(Deserialize, Debug)]
struct Term {
    uri: String,
    kenmerk: Kenmerk,
}
#[derive(Deserialize, Debug)]
struct Kenmerk {
    naam: String,
}

/// One line of `provenance.jsonl`.
#[derive(Serialize)]
struct Record {
    image_id: u32,
    object: ErfgoedObject,
    typologie: Vec<String>,
}

fn typologies(detail: Detail) -> Vec<String> {
    detail
        .kenmerkgroepen
        .into_iter()
        .flat_map(|g| g.thesaurus)
        .filter(|t| t.kenmerk.naam == "typologie")
        .map(|t| t.uri)
        .collect()
}

fn buildings<'a>(
    client: &'a UnstirClient,
    base: &'a Url,
) -> impl Stream<Item = anyhow::Result<ErfgoedObject>> + 'a {
    stream::try_unfold(Some(0usize), move |start| async move {
        let Some(start) = start else {
            return anyhow::Ok(None);
        };
        let page = client
            .get(base.join("erfgoedobjecten")?)
            .query(&[("typologie", format!("{TYPES}{ROOT}"))])
            .query(&FILTERS)
            .header("Range", format!("items={}-{}", start, start + PAGE - 1))
            .json::<Vec<ErfgoedObject>>()
            .await?;

        if page.is_empty() {
            return Ok(None);
        }

        let next = (page.len() == PAGE).then_some(start + PAGE);
        Ok(Some((stream::iter(page.into_iter().map(anyhow::Ok)), next)))
    })
    .try_flatten()
}

/// What happened to one object.
enum Outcome {
    /// No primary photo.
    NoPhoto,
    /// Photo on disk and in `provenance.jsonl` already.
    Known,
    New(Record),
    Failed { object: u32, error: String },
}

/// One line of `failures.jsonl`.
#[derive(Serialize)]
struct Failure<'a> {
    object: u32,
    error: &'a str,
}

struct Paths {
    images: PathBuf,
    dithered: PathBuf,
}

/// The image ids in `provenance.jsonl`. A last line without its newline (a run
/// that stopped while writing it) is left out, and gets its newline, so the
/// next line starts on its own.
fn known_images(path: &Path) -> anyhow::Result<HashSet<u32>> {
    #[derive(Deserialize)]
    struct Seen {
        image_id: u32,
    }

    let mut file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)?;
    let known = BufReader::new(&file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| serde_json::from_str::<Seen>(&line).ok())
        .map(|seen| seen.image_id)
        .collect();

    if file.metadata()?.len() > 0 {
        let mut last = [0u8];
        file.seek(SeekFrom::End(-1))?;
        file.read_exact(&mut last)?;
        if last[0] != b'\n' {
            file.write_all(b"\n")?;
        }
    }
    Ok(known)
}

fn append(path: &Path) -> anyhow::Result<File> {
    Ok(OpenOptions::new().append(true).create(true).open(path)?)
}

/// The photo of `image_id`, as `<id>.jpg` and its dithered `styled/<id>.png`.
/// The jpg is written last, through a temporary file, so a jpg on disk is whole.
async fn download(client: &UnstirClient, image_id: u32, paths: &Paths) -> anyhow::Result<()> {
    let bytes = client
        .get(format!("{BASE_IMAGES}/images/{image_id}/content/full"))
        .bytes()
        .await?;
    let dithered = paths.dithered.join(format!("{image_id}.png"));
    let bytes = tokio::task::spawn_blocking(move || -> anyhow::Result<bytes::Bytes> {
        print_image_bytes(&bytes)?.save(dithered)?;
        Ok(bytes)
    })
    .await??;
    let image = paths.images.join(format!("{image_id}.jpg"));
    let partial = paths.images.join(format!("{image_id}.jpg.part"));
    tokio::fs::write(&partial, &bytes).await?;
    tokio::fs::rename(&partial, &image).await?;
    Ok(())
}

/// Download what is missing of `object`: its photo when it is not on disk, its
/// typologies when it is not in `provenance.jsonl`.
async fn fetch(
    client: &UnstirClient,
    base: &Url,
    paths: &Paths,
    known: &HashSet<u32>,
    object: ErfgoedObject,
) -> anyhow::Result<Outcome> {
    let Some(image_id) = object
        .primaire_foto
        .as_deref()
        .and_then(|f| f.rsplit('/').next()?.parse::<u32>().ok())
    else {
        return Ok(Outcome::NoPhoto);
    };

    let on_disk = tokio::fs::try_exists(paths.images.join(format!("{image_id}.jpg"))).await?;
    if !on_disk {
        download(client, image_id, paths).await?;
    }
    if known.contains(&image_id) {
        return Ok(Outcome::Known);
    }

    let detail = client
        .get(base.join(&format!("erfgoedobjecten/{}", object.id))?)
        .json::<Detail>()
        .await?;
    Ok(Outcome::New(Record {
        image_id,
        object,
        typologie: typologies(detail),
    }))
}

#[derive(Default, Debug)]
struct Counts {
    new: usize,
    known: usize,
    no_photo: usize,
    failed: usize,
}

struct Output {
    provenance: File,
    failures: File,
    counts: Counts,
}

fn line<T: Serialize>(value: &T) -> anyhow::Result<Vec<u8>> {
    let mut line = serde_json::to_vec(value)?;
    line.push(b'\n');
    Ok(line)
}

impl Output {
    fn write(&mut self, outcome: Outcome) -> anyhow::Result<()> {
        match outcome {
            Outcome::NoPhoto => self.counts.no_photo += 1,
            Outcome::Known => self.counts.known += 1,
            Outcome::New(record) => {
                self.provenance.write_all(&line(&record)?)?;
                self.counts.new += 1;
            }
            Outcome::Failed { object, error } => {
                eprintln!("FAILED object {object}: {error}");
                let failure = Failure {
                    object,
                    error: &error,
                };
                self.failures.write_all(&line(&failure)?)?;
                self.counts.failed += 1;
            }
        }
        Ok(())
    }
}

/// Download the primary photo and provenance of every building, into `./imgs`.
///
/// A run continues where an earlier one stopped: photos on disk are not
/// downloaded again, and objects in `provenance.jsonl` are not asked for again.
/// An object that fails (after retries) is logged in `failures.jsonl`, and the
/// run goes on. Only a page of the object list that keeps failing stops it.
pub async fn get_buildings(client: &UnstirClient) -> anyhow::Result<()> {
    let images = PathBuf::from("./imgs");
    let paths = Paths {
        dithered: images.join("styled"),
        images,
    };
    tokio::fs::create_dir_all(&paths.dithered).await?;
    let provenance_path = paths.images.join("provenance.jsonl");
    let known = known_images(&provenance_path)?;
    eprintln!(
        "{} objects already in {}",
        known.len(),
        provenance_path.display()
    );
    let output = Output {
        provenance: append(&provenance_path)?,
        failures: append(&paths.images.join("failures.jsonl"))?,
        counts: Counts::default(),
    };

    let base = Url::parse(BASE_INVENTORY)?;
    let (paths, known, base) = (&paths, &known, &base);

    let output = buildings(client, base)
        .map_ok(|object| async move {
            let id = object.id;
            anyhow::Ok(
                fetch(client, base, paths, known, object)
                    .await
                    .unwrap_or_else(|error| Outcome::Failed {
                        object: id,
                        error: format!("{error:#}"),
                    }),
            )
        })
        .try_buffer_unordered(8)
        .try_fold(output, |mut output, outcome| async move {
            tokio::task::spawn_blocking(move || {
                output.write(outcome)?;
                anyhow::Ok(output)
            })
            .await?
        })
        .await?;

    eprintln!("done: {:?}", output.counts);
    Ok(())
}
