use std::{io::Write as _, path::PathBuf};

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

pub async fn get_buildings(client: &UnstirClient) -> anyhow::Result<()> {
    let images_path = PathBuf::from("./imgs");
    let dithered_path = images_path.join("styled");
    tokio::fs::create_dir_all(&dithered_path).await?;
    let provenance = std::fs::File::create(images_path.join("provenance.jsonl"))?;

    let base = Url::parse(BASE_INVENTORY)?;
    let (images_path, dithered_path, base) = (&images_path, &dithered_path, &base);

    buildings(client, base)
        .map_ok(|object| async move {
            let Some(image_id) = object
                .primaire_foto
                .as_deref()
                .and_then(|f| f.rsplit('/').next()?.parse::<u32>().ok())
            else {
                return anyhow::Ok(None);
            };

            let detail = client
                .get(base.join(&format!("erfgoedobjecten/{}", object.id))?)
                .json::<Detail>()
                .await?;

            Ok(Some(Record {
                image_id,
                object,
                typologie: typologies(detail),
            }))
        })
        .try_buffer_unordered(4)
        .try_filter_map(|record| async move { Ok(record) })
        .map_ok(|record| async move {
            let bytes = client
                .get(format!("{BASE_IMAGES}/images/{}/content/full", record.image_id))
                .bytes()
                .await?;
            tokio::fs::write(images_path.join(format!("{}.jpg", record.image_id)), &bytes).await?;
            let out_dithered = dithered_path.join(format!("{}.png", record.image_id));
            tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
                print_image_bytes(&bytes)?.save(out_dithered)?;
                Ok(())
            })
            .await??;
            anyhow::Ok(record)
        })
        .try_buffer_unordered(4)
        .try_fold(provenance, |mut file, record| async move {
            tokio::task::spawn_blocking(move || {
                let mut line = serde_json::to_vec(&record)?;
                line.push(b'\n');
                file.write_all(&line)?;
                anyhow::Ok(file)
            })
            .await?
        })
        .await?;
    Ok(())
}
