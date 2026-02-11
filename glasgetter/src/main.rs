use std::{io::stdout, path::PathBuf};

use clap::{Parser, Subcommand};
use futures::{Stream, StreamExt, stream};
use indicatif::ProgressStyle;
use reqwest::Client;
use scraper::{Html, Selector};

use crate::model::{Artist, Artwork};

const BASE_URL: &str = "https://www.wikiart.org";

mod model;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List all artworks on wikiart
    ListArtworks {
        #[arg(short, long, default_value_t = 10)]
        /// The amount of in-flight requests to allow
        buffer: usize,

        #[arg(short, long)]
        /// The output file to write the json to.
        /// If not given, writes to stdout
        output: Option<PathBuf>,
    },

    /// Download all images from wikiart
    DownloadImages {
        #[arg(short, long, default_value_t = 10)]
        /// The amount of in-flight requests to allow
        buffer: usize,

        #[arg(short, long)]
        /// The input file with artworks json
        input: PathBuf,

        #[arg(short, long)]
        /// The output directory to save images to
        output: PathBuf,
    },
}

#[inline]
fn wiki_url(resource: &str) -> String {
    match resource.chars().next().is_some_and(|c| c == '/') {
        true => format!("{BASE_URL}{resource}"),
        false => format!("{BASE_URL}/{resource}"),
    }
}

async fn list_works_by(client: &Client, artist: Artist) -> Vec<model::Artwork> {
    let response_html = client
        .get(format!(
            "{}/all-works/text-list",
            wiki_url(&artist.resource)
        ))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let doc = Html::parse_document(&response_html);
    let sel = Selector::parse(".painting-list-text-row > a[href]").unwrap();
    doc.select(&sel)
        .filter_map(|el| {
            let resource = String::from(el.value().attr("href")?);
            let name = el.inner_html();
            Some(model::Artwork {
                name,
                resource,
                artist_name: artist.name.clone(),
                artist_resource: artist.resource.clone(),
            })
        })
        .collect()
}

async fn get_work_image_url(client: &Client, artwork: &Artwork) -> Option<String> {
    // wiki-layout-artist-image-wrapper
    let response_html = client
        .get(wiki_url(&artwork.resource))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let doc = Html::parse_document(&response_html);
    let sel = Selector::parse(".wiki-layout-artist-image-wrapper > img[src]").unwrap();
    let src = doc
        .select(&sel)
        .next()
        .and_then(|el| el.value().attr("src").map(|e| e.to_string()))?;

    Some(match src.split_once('!') {
        Some((src, _)) => src.to_string(),
        None => src,
    })
}

async fn list_artist_starting_with_letter(client: &Client, letter: char) -> Vec<model::Artist> {
    let response_html = client
        .get(wiki_url(&format!("/en/Alphabet/{letter}/text-list")))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let doc = Html::parse_document(&response_html);
    let sel = Selector::parse(".masonry-text-view-all li").unwrap();
    let link_sel = Selector::parse("a[href]").unwrap();
    let span_sel = Selector::parse("span").unwrap();
    doc.select(&sel)
        .filter_map(|li| {
            let a = li.select(&link_sel).next()?;
            let resource = String::from(a.value().attr("href")?);
            let name = a.inner_html();
            let artwork_count = li
                .select(&span_sel)
                .map(|s| s.inner_html())
                .find(|text| text.contains("artwork"))?
                .chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(0);
            Some(model::Artist {
                name,
                resource,
                artwork_count,
            })
        })
        .collect()
}

fn list_all_artists<'a>(client: &'a Client) -> impl Stream<Item = model::Artist> + 'a {
    stream::iter('a'..='z')
        .map(move |letter| list_artist_starting_with_letter(client, letter))
        .buffer_unordered(10)
        .flat_map(stream::iter)
}

async fn list_all_artworks(buffer: usize) -> Vec<Artwork> {
    let client = Client::new();

    let artists: Vec<_> = list_all_artists(&client).collect().await;
    let total_artworks: u64 = artists.iter().map(|a| a.artwork_count).sum();
    let pb = indicatif::ProgressBar::new(total_artworks);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40} {pos}/{len} artworks [{elapsed_precise} / {eta_precise}]")
            .unwrap(),
    );

    let artworks: Vec<Artwork> = stream::iter(artists)
        .map(|artist| list_works_by(&client, artist))
        .buffer_unordered(buffer)
        .flat_map(stream::iter)
        .map(|work| {
            pb.inc(1);
            let pb = pb.clone();
            async move {
                pb.println(format!("{work}"));
                work
            }
        })
        .buffered(1)
        .collect()
        .await;

    pb.finish_with_message("All artworks found.");

    artworks
}

async fn write_all_artworks(buffer: usize, output: Option<PathBuf>) {
    let artworks = list_all_artworks(buffer).await;

    if let Some(output_file) = output {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(output_file)
            .unwrap();
        serde_json::to_writer(&mut file, &artworks).unwrap();
    } else {
        serde_json::to_writer(stdout(), &artworks).unwrap();
    }
}

fn image_exists(output_dir: &PathBuf, filename: &str) -> bool {
    for ext in ["jpg", "jpeg", "png", "gif", "webp"] {
        if output_dir.join(format!("{filename}.{ext}")).exists() {
            return true;
        }
    }
    false
}

async fn download_image(client: &Client, artwork: &Artwork, output_dir: &PathBuf) {
    let filename = artwork
        .resource
        .trim_start_matches('/')
        .trim_start_matches("en/")
        .replace('/', "_")
        .to_lowercase();

    if image_exists(output_dir, &filename) {
        return;
    }

    let Some(image_url) = get_work_image_url(client, artwork).await else {
        return;
    };

    let extension = image_url
        .rsplit('.')
        .next()
        .unwrap_or("jpg");

    let filepath = output_dir.join(format!("{filename}.{extension}"));

    let bytes = client.get(&image_url).send().await.unwrap().bytes().await.unwrap();
    std::fs::write(&filepath, &bytes).unwrap();
}

async fn download_all_images(buffer: usize, input: PathBuf, output: PathBuf) {
    let client = Client::new();

    std::fs::create_dir_all(&output).unwrap();

    let artworks: Vec<Artwork> = serde_json::from_reader(std::fs::File::open(&input).unwrap()).unwrap();
    let total = artworks.len() as u64;

    let pb = indicatif::ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40} {pos}/{len} images [{elapsed_precise} / {eta_precise}]")
            .unwrap(),
    );

    stream::iter(artworks)
        .map(|artwork| {
            let client = &client;
            let output = &output;
            async move {
                download_image(client, &artwork, output).await;
            }
        })
        .buffer_unordered(buffer)
        .for_each(|_| {
            pb.inc(1);
            async {}
        })
        .await;

    pb.finish_with_message("All images downloaded.");
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args = Args::parse();

    match args.command {
        Command::ListArtworks { buffer, output } => write_all_artworks(buffer, output).await,
        Command::DownloadImages { buffer, input, output } => {
            download_all_images(buffer, input, output).await
        }
    }
}
