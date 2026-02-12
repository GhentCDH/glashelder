use async_trait::async_trait;
use futures::{Stream, StreamExt, stream};
use reqwest::Client;
use scraper::{Html, Selector};

use crate::client::fetch_html;
use crate::error::Result;
use crate::model::{BalatEntry, UnifiedArtwork};

use super::ArtSource;

const BASE_URL: &str = "https://balat.kikirpa.be";
const DEFAULT_ROLE: &str = "schilder";
const DEFAULT_LIMIT: u32 = 240;

/// BALAT (Belgian Art Links and Tools) artwork source
pub struct BalatSource {
    /// Creator role to filter by (e.g., "schilder" for painters)
    pub role: String,
    /// Results per page
    pub limit: u32,
}

impl BalatSource {
    pub fn new() -> Self {
        BalatSource {
            role: DEFAULT_ROLE.to_string(),
            limit: DEFAULT_LIMIT,
        }
    }

    pub fn with_role(mut self, role: &str) -> Self {
        self.role = role.to_string();
        self
    }

    pub fn with_limit(mut self, limit: u32) -> Self {
        self.limit = limit;
        self
    }

    /// Build the initial search URL (with full query parameters)
    fn initial_search_url(&self) -> String {
        format!(
            "{BASE_URL}/results.php?typesearch=advanced&rolecreator={}&style=col4&limit={}&sort=unsorted",
            self.role, self.limit
        )
    }

    /// Build a pagination URL (query is saved in session cookie)
    fn pagination_url(&self, start_from: u32) -> String {
        format!("{BASE_URL}/results.php?startfrom={start_from}")
    }
}

impl Default for BalatSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ArtSource for BalatSource {
    fn name(&self) -> &'static str {
        "BALAT"
    }

    fn base_url(&self) -> &str {
        BASE_URL
    }

    fn list_artworks<'a>(
        &'a self,
        client: &'a Client,
    ) -> Box<dyn Stream<Item = Result<UnifiedArtwork>> + Send + Unpin + 'a> {
        Box::new(Box::pin(list_all_artworks_stream(self, client)))
    }

    async fn get_image_url(
        &self,
        _client: &Client,
        artwork: &UnifiedArtwork,
    ) -> Result<Option<String>> {
        // BALAT artworks already have their image URL set
        Ok(artwork.image_url.clone())
    }
}

/// Parse total results count from the info box
/// Format: <div class="info-box">... results:<strong> 82473</strong>)</div>
fn parse_total_results(html: &str) -> Option<u32> {
    let doc = Html::parse_document(html);
    let info_sel = Selector::parse("div.info-box strong").ok()?;
    doc.select(&info_sel)
        .next()
        .and_then(|el| el.text().collect::<String>().trim().parse().ok())
}

/// Parse BALAT search results page
fn parse_search_results(html: &str) -> Result<Vec<BalatEntry>> {
    let doc = Html::parse_document(html);

    // Select list items in the results
    let li_sel = Selector::parse("li.item.artwork").expect("static selector should be valid");
    let img_sel =
        Selector::parse("img[src*='iiif.kikirpa.be']").expect("static selector should be valid");
    let link_sel = Selector::parse("h5 > a[href]").expect("static selector should be valid");
    let creator_sel = Selector::parse("strong").expect("static selector should be valid");

    let mut entries = Vec::new();

    for li in doc.select(&li_sel) {
        // Get IIIF image URL
        let Some(img) = li.select(&img_sel).next() else {
            continue;
        };
        let Some(thumbnail_url) = img.value().attr("src").map(String::from) else {
            continue;
        };

        // Extract ID from IIIF URL
        // Format: https://iiif.kikirpa.be/iiif/2/[ID]/full/!280,280/0/default.jpg
        let id = thumbnail_url
            .split("/iiif/2/")
            .nth(1)
            .and_then(|s| s.split('/').next())
            .unwrap_or("")
            .to_string();

        if id.is_empty() {
            continue;
        }

        // Get title and detail link
        let (title, detail_path) = if let Some(a) = li.select(&link_sel).next() {
            let href = a.value().attr("href").unwrap_or("").to_string();
            let text = a.text().collect::<String>().trim().to_string();
            // Strip number prefix like "1. " or "123. "
            let title = text
                .strip_prefix(|c: char| c.is_ascii_digit())
                .map(|s| {
                    s.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.')
                        .trim_start()
                })
                .unwrap_or(&text)
                .to_string();
            (title, href)
        } else {
            continue;
        };

        // Get creator name (first <strong> element)
        let creator_name = li
            .select(&creator_sel)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_string())
            .filter(|s| !s.is_empty());

        entries.push(BalatEntry {
            id,
            title,
            detail_path,
            creator_name,
            thumbnail_url,
        });
    }

    Ok(entries)
}

/// Initialize session by hitting the suggest endpoint with the search term
async fn init_session(client: &Client, role: &str) -> Result<()> {
    // Hit the suggest endpoint to establish the search session
    let url = format!("{BASE_URL}/suggest_photo.php?field=creatorrole&term={role}");
    let _ = fetch_html(client, &url).await?;
    Ok(())
}

/// Fetch first page and return entries + total count
async fn fetch_first_page(
    source: &BalatSource,
    client: &Client,
) -> Result<(Vec<BalatEntry>, u32)> {
    // Initialize session
    init_session(client, &source.role).await?;

    let url = source.initial_search_url();
    eprintln!("[BALAT] Fetching first page: {url}");
    let html = fetch_html(client, &url).await?;

    let total = parse_total_results(&html).unwrap_or(0);
    let entries = parse_search_results(&html)?;
    eprintln!(
        "[BALAT] First page: {} entries, total results: {}",
        entries.len(),
        total
    );

    Ok((entries, total))
}

/// Fetch a single page of BALAT results by page number
async fn fetch_page(source: &BalatSource, client: &Client, page: u32) -> Result<Vec<BalatEntry>> {
    let url = source.pagination_url(page);
    let html = fetch_html(client, &url).await?;
    let entries = parse_search_results(&html)?;
    eprintln!("[BALAT] Page {}: {} entries", page, entries.len());
    Ok(entries)
}

/// Stream all artworks from BALAT using parallel pagination
fn list_all_artworks_stream<'a>(
    source: &'a BalatSource,
    client: &'a Client,
) -> impl Stream<Item = Result<UnifiedArtwork>> + Send + 'a {
    // Use once + flatten to handle async setup, then stream pages in parallel
    stream::once(async move {
        // Fetch first page to get total and establish session
        let (first_entries, total) = match fetch_first_page(source, client).await {
            Ok(result) => result,
            Err(e) => {
                eprintln!("[BALAT] Error fetching first page: {e}");
                return stream::empty().boxed();
            }
        };

        let total_pages = (total + source.limit - 1) / source.limit;
        eprintln!(
            "[BALAT] Total: {} results, {} pages (limit={})",
            total, total_pages, source.limit
        );

        // Stream first page entries first
        let first_stream = stream::iter(first_entries);

        // Then stream remaining pages in parallel, yielding entries as each page completes
        let remaining_stream = stream::iter(2..=total_pages)
            .map(move |page| async move { fetch_page(source, client, page).await })
            .buffer_unordered(10)
            .flat_map(|result| match result {
                Ok(entries) => stream::iter(entries).boxed(),
                Err(e) => {
                    eprintln!("[BALAT] Error fetching page: {e}");
                    stream::empty().boxed()
                }
            });

        first_stream.chain(remaining_stream).boxed()
    })
    .flatten()
    .map(|entry| Ok(UnifiedArtwork::from(entry)))
}
