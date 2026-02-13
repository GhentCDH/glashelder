use std::{
    io::stdout,
    path::{Path, PathBuf},
};

use clap::{Parser, Subcommand};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;

mod client;
mod error;
mod model;
mod source;

use client::{build_client, fetch_bytes};
use error::{Error, Result};
use model::{Source, UnifiedArtwork};
use source::{ArtSource, BalatSource, WikiArtSource};

#[derive(Parser)]
#[command(version, about = "Multi-source artwork scraper", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// WikiArt commands
    #[command(subcommand)]
    Wikiart(WikiartCommand),

    /// BALAT commands
    #[command(subcommand)]
    Balat(BalatCommand),

    /// Download images from any source (auto-detects from JSON)
    Download {
        #[arg(short, long, default_value_t = 10)]
        /// The amount of in-flight requests to allow
        buffer: usize,

        #[arg(short, long)]
        /// The input file with artworks JSON
        input: PathBuf,

        #[arg(short, long)]
        /// The output directory to save images to
        output: PathBuf,
    },
}

#[derive(Subcommand)]
enum WikiartCommand {
    /// List all artworks on WikiArt
    List {
        #[arg(short, long, default_value_t = 10)]
        /// The amount of in-flight requests to allow
        buffer: usize,

        #[arg(short, long)]
        /// The output file to write the JSON to.
        /// If not given, writes to stdout
        output: Option<PathBuf>,
    },

    /// Download images from WikiArt
    Download {
        #[arg(short, long, default_value_t = 10)]
        /// The amount of in-flight requests to allow
        buffer: usize,

        #[arg(short, long)]
        /// The input file with artworks JSON
        input: PathBuf,

        #[arg(short, long)]
        /// The output directory to save images to
        output: PathBuf,
    },
}

#[derive(Subcommand)]
enum BalatCommand {
    /// List artworks from BALAT
    List {
        #[arg(short, long, default_value = "schilder")]
        /// Creator role to filter by (e.g., "schilder" for painters)
        role: String,

        #[arg(short, long, default_value_t = 240)]
        /// Results per page
        limit: u32,

        #[arg(short, long)]
        /// Maximum number of artworks to fetch (0 = unlimited)
        max: Option<usize>,

        #[arg(short, long)]
        /// The output file to write the JSON to.
        /// If not given, writes to stdout
        output: Option<PathBuf>,
    },

    /// Download images from BALAT
    Download {
        #[arg(short, long, default_value_t = 10)]
        /// The amount of in-flight requests to allow
        buffer: usize,

        #[arg(short, long)]
        /// The input file with artworks JSON
        input: PathBuf,

        #[arg(short, long)]
        /// The output directory to save images to
        output: PathBuf,
    },
}

// Helper to create progress bar
fn create_progress_bar(total: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "{{bar:40}} {{pos}}/{{len}} {message} [{{elapsed_precise}} / {{eta_precise}}]"
            ))
            .expect("static template should be valid"),
    );
    pb
}

// WikiArt list command
async fn wikiart_list(_buffer: usize, output: Option<PathBuf>) -> Result<()> {
    let client = build_client()?;
    let source = WikiArtSource::new();

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner} {msg}")
            .expect("static template should be valid"),
    );
    pb.set_message("Fetching artworks from WikiArt...");
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let mut artworks: Vec<UnifiedArtwork> = Vec::new();
    let mut stream = source.list_artworks(&client);

    // Buffer artworks for progress display
    let mut count = 0u64;
    while let Some(result) = stream.next().await {
        match result {
            Ok(artwork) => {
                artworks.push(artwork);
                count += 1;
                if count.is_multiple_of(100) {
                    pb.set_message(format!("Fetched {count} artworks..."));
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
            }
        }
    }

    pb.finish_with_message(format!("Found {count} artworks"));

    write_artworks(&artworks, output)?;
    Ok(())
}

// BALAT list command
async fn balat_list(
    role: String,
    limit: u32,
    max: Option<usize>,
    output: Option<PathBuf>,
) -> Result<()> {
    let client = build_client()?;
    let source = BalatSource::new().with_role(&role).with_limit(limit);

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner} {msg}")
            .expect("static template should be valid"),
    );
    pb.set_message("Fetching artworks from BALAT...");
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let mut artworks: Vec<UnifiedArtwork> = Vec::new();
    let mut stream = source.list_artworks(&client);

    let mut count = 0u64;
    while let Some(result) = stream.next().await {
        match result {
            Ok(artwork) => {
                artworks.push(artwork);
                count += 1;
                if count.is_multiple_of(100) {
                    pb.set_message(format!("Fetched {count} artworks..."));
                }

                // Check max limit
                if let Some(m) = max
                    && artworks.len() >= m
                {
                    break;
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
            }
        }
    }

    pb.finish_with_message(format!("Found {count} artworks"));

    write_artworks(&artworks, output)?;
    Ok(())
}

// Write artworks to file or stdout
fn write_artworks(artworks: &[UnifiedArtwork], output: Option<PathBuf>) -> Result<()> {
    if let Some(output_file) = output {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&output_file)?;
        serde_json::to_writer(file, artworks)?;
        eprintln!(
            "wrote {} artworks to {}",
            artworks.len(),
            output_file.display()
        );
    } else {
        serde_json::to_writer(stdout(), artworks)?;
    }
    Ok(())
}

// Check if image already exists
fn image_exists(output_dir: &Path, filename: &str) -> bool {
    for ext in ["jpg", "jpeg", "png", "gif", "webp"] {
        if output_dir.join(format!("{filename}.{ext}")).exists() {
            return true;
        }
    }
    false
}

// Generate filename from artwork
fn artwork_filename(artwork: &UnifiedArtwork) -> String {
    match artwork.source {
        Source::WikiArt => artwork
            .resource
            .trim_start_matches('/')
            .trim_start_matches("en/")
            .replace('/', "_")
            .to_lowercase(),
        Source::Balat => format!("balat_{}", artwork.id),
    }
}

// Download a single image
async fn download_image(
    client: &Client,
    source: &dyn ArtSource,
    artwork: &UnifiedArtwork,
    output_dir: &Path,
    pb: &ProgressBar,
) -> Result<()> {
    let filename = artwork_filename(artwork);

    if image_exists(output_dir, &filename) {
        return Ok(());
    }

    let image_url = match source.get_image_url(client, artwork).await? {
        Some(url) => url,
        None => {
            pb.println(format!("warning: no image found for {}", artwork.title));
            return Ok(());
        }
    };

    let extension = image_url.rsplit('.').next().unwrap_or("jpg");
    let filepath = output_dir.join(format!("{filename}.{extension}"));

    let bytes = fetch_bytes(client, &image_url).await?;
    std::fs::write(&filepath, &bytes)?;

    Ok(())
}

// Download images for a specific source
async fn download_images_with_source(
    source: &dyn ArtSource,
    buffer: usize,
    input: PathBuf,
    output: PathBuf,
) -> Result<()> {
    let client = build_client()?;

    std::fs::create_dir_all(&output)?;

    let file = std::fs::File::open(&input)?;
    let artworks: Vec<UnifiedArtwork> = serde_json::from_reader(file)?;
    let total = artworks.len() as u64;

    let pb = create_progress_bar(total, "images");

    let errors = std::sync::atomic::AtomicU64::new(0);
    let restricted = std::sync::atomic::AtomicU64::new(0);

    futures::stream::iter(artworks)
        .map(|artwork| {
            let client = &client;
            let output = &output;
            let pb = &pb;
            let errors = &errors;
            let restricted = &restricted;
            async move {
                if let Err(e) = download_image(client, source, &artwork, output, pb).await {
                    pb.println(format!("error downloading {}: {e}", artwork.title));
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

// Unified download that auto-detects source from JSON
async fn download_unified(buffer: usize, input: PathBuf, output: PathBuf) -> Result<()> {
    let client = build_client()?;

    std::fs::create_dir_all(&output)?;

    let file = std::fs::File::open(&input)?;
    let artworks: Vec<UnifiedArtwork> = serde_json::from_reader(file)?;

    if artworks.is_empty() {
        eprintln!("no artworks found in input file");
        return Ok(());
    }

    let total = artworks.len() as u64;
    let pb = create_progress_bar(total, "images");

    let wikiart_source = WikiArtSource::new();
    let balat_source = BalatSource::new();

    let errors = std::sync::atomic::AtomicU64::new(0);
    let restricted = std::sync::atomic::AtomicU64::new(0);

    futures::stream::iter(artworks)
        .map(|artwork| {
            let client = &client;
            let output = &output;
            let pb = &pb;
            let errors = &errors;
            let restricted = &restricted;
            let wikiart_source = &wikiart_source;
            let balat_source = &balat_source;
            async move {
                let source: &dyn ArtSource = match artwork.source {
                    Source::WikiArt => wikiart_source,
                    Source::Balat => balat_source,
                };

                if let Err(e) = download_image(client, source, &artwork, output, pb).await {
                    pb.println(format!("error downloading {}: {e}", artwork.title));
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
        Command::Wikiart(cmd) => match cmd {
            WikiartCommand::List { buffer, output } => wikiart_list(buffer, output).await,
            WikiartCommand::Download {
                buffer,
                input,
                output,
            } => {
                let source = WikiArtSource::new();
                download_images_with_source(&source, buffer, input, output).await
            }
        },
        Command::Balat(cmd) => match cmd {
            BalatCommand::List {
                role,
                limit,
                max,
                output,
            } => balat_list(role, limit, max, output).await,
            BalatCommand::Download {
                buffer,
                input,
                output,
            } => {
                let source = BalatSource::new();
                download_images_with_source(&source, buffer, input, output).await
            }
        },
        Command::Download {
            buffer,
            input,
            output,
        } => download_unified(buffer, input, output).await,
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
