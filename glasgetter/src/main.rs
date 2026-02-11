use std::{
    io::stdout,
    path::{Path, PathBuf},
};

use clap::{Parser, Subcommand};
use futures::{Stream, StreamExt, stream};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;

mod client;
mod error;
mod model;
mod scraper;

use client::{build_client, fetch_bytes};
use error::Result;
use model::{Artist, Artwork};
use scraper::{get_artwork_image_url, list_artists_by_letter, list_works_by_artist};

use crate::error::Error;

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

fn list_all_artists(client: &Client) -> impl Stream<Item = Result<Artist>> + '_ {
    stream::iter('a'..='z')
        .map(move |letter| async move { list_artists_by_letter(client, letter).await })
        .buffer_unordered(10)
        .flat_map(|result| match result {
            Ok(artists) => stream::iter(artists.into_iter().map(Ok).collect::<Vec<_>>()),
            Err(e) => {
                eprintln!("error fetching artists: {e}");
                stream::iter(vec![])
            }
        })
}

async fn list_all_artworks(client: &Client, buffer: usize, pb: &ProgressBar) -> Vec<Artwork> {
    let artists: Vec<_> = list_all_artists(client)
        .filter_map(|r| async { r.ok() })
        .collect()
        .await;

    let total_artworks: u64 = artists.iter().map(|a| a.artwork_count).sum();
    pb.set_length(total_artworks);

    stream::iter(artists)
        .map(|artist| async move {
            match list_works_by_artist(client, &artist).await {
                Ok(works) => works,
                Err(e) => {
                    eprintln!("error fetching works for {}: {e}", artist.name);
                    vec![]
                }
            }
        })
        .buffer_unordered(buffer)
        .flat_map(stream::iter)
        .map(|work| {
            pb.inc(1);
            work
        })
        .collect()
        .await
}

async fn write_all_artworks(buffer: usize, output: Option<PathBuf>) -> Result<()> {
    let client = build_client()?;

    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40} {pos}/{len} artworks [{elapsed_precise} / {eta_precise}]")
            .expect("static template should be valid"),
    );

    let artworks = list_all_artworks(&client, buffer, &pb).await;
    pb.finish_with_message("all artworks found.");

    if let Some(output_file) = output {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&output_file)?;
        serde_json::to_writer(file, &artworks)?;
        eprintln!(
            "wrote {} artworks to {}",
            artworks.len(),
            output_file.display()
        );
    } else {
        serde_json::to_writer(stdout(), &artworks)?;
    }

    Ok(())
}

fn image_exists(output_dir: &Path, filename: &str) -> bool {
    for ext in ["jpg", "jpeg", "png", "gif", "webp"] {
        if output_dir.join(format!("{filename}.{ext}")).exists() {
            return true;
        }
    }
    false
}

async fn download_image(
    client: &Client,
    artwork: &Artwork,
    output_dir: &Path,
    pb: &ProgressBar,
) -> Result<()> {
    let filename = artwork
        .resource
        .trim_start_matches('/')
        .trim_start_matches("en/")
        .replace('/', "_")
        .to_lowercase();

    if image_exists(output_dir, &filename) {
        return Ok(());
    }

    let image_url = match get_artwork_image_url(client, artwork).await? {
        Some(url) => url,
        None => {
            pb.println(format!("warning: no image found for {}", artwork.name));
            return Ok(());
        }
    };

    let extension = image_url.rsplit('.').next().unwrap_or("jpg");
    let filepath = output_dir.join(format!("{filename}.{extension}"));

    let bytes = fetch_bytes(client, &image_url).await?;
    std::fs::write(&filepath, &bytes)?;

    Ok(())
}

async fn download_all_images(buffer: usize, input: PathBuf, output: PathBuf) -> Result<()> {
    let client = build_client()?;

    std::fs::create_dir_all(&output)?;

    let file = std::fs::File::open(&input)?;
    let artworks: Vec<Artwork> = serde_json::from_reader(file)?;
    let total = artworks.len() as u64;

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40} {pos}/{len} images [{elapsed_precise} / {eta_precise}]")
            .expect("static template should be valid"),
    );

    let errors = std::sync::atomic::AtomicU64::new(0);
    let restricted = std::sync::atomic::AtomicU64::new(0);

    stream::iter(artworks)
        .map(|artwork| {
            let client = &client;
            let output = &output;
            let pb = &pb;
            let errors = &errors;
            let restricted = &restricted;
            async move {
                if let Err(e) = download_image(client, &artwork, output, pb).await {
                    pb.println(format!("error downloading {}: {e}", artwork.name));
                    if let Error::Restricted = e {
                        restricted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        })
        .buffer_unordered(buffer)
        .for_each(|_| {
            pb.inc(1);
            async {}
        })
        .await;

    let error_count = errors.load(std::sync::atomic::Ordering::Relaxed);
    if error_count > 0 {
        pb.finish_with_message(format!("done with {error_count} errors."));
    } else {
        pb.finish_with_message("all images downloaded.");
    }

    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args = Args::parse();

    let result = match args.command {
        Command::ListArtworks { buffer, output } => write_all_artworks(buffer, output).await,
        Command::DownloadImages {
            buffer,
            input,
            output,
        } => download_all_images(buffer, input, output).await,
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
