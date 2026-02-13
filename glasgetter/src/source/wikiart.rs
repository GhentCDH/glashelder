use async_trait::async_trait;
use futures::{Stream, StreamExt, stream};
use reqwest::Client;
use scraper::{Html, Selector};

use crate::client::fetch_html;
use crate::error::{Error, Result};
use crate::model::{UnifiedArtwork, WikiArtist, WikiArtwork};

use super::ArtSource;

const BASE_URL: &str = "https://www.wikiart.org";

/// WikiArt artwork source
pub struct WikiArtSource;

impl WikiArtSource {
    pub fn new() -> Self {
        WikiArtSource
    }

    /// Construct a full WikiArt URL from a resource path
    pub fn url(&self, resource: &str) -> String {
        // The resource possibly isn't a relative path, if it refers to an external resource
        if resource.contains("http:") {
            return String::from(resource);
        }

        match resource.chars().next().is_some_and(|c| c == '/') {
            true => format!("{BASE_URL}{resource}"),
            false => format!("{BASE_URL}/{resource}"),
        }
    }
}

impl Default for WikiArtSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ArtSource for WikiArtSource {
    fn name(&self) -> &'static str {
        "WikiArt"
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
        client: &Client,
        artwork: &UnifiedArtwork,
    ) -> Result<Option<String>> {
        let url = self.url(&artwork.resource);
        let html = fetch_html(client, &url).await?;
        parse_artwork_image_url(&html)
    }
}

/// Parse artists from the alphabet listing page HTML
fn parse_artists(html: &str, url: &str) -> Result<Vec<WikiArtist>> {
    let doc = Html::parse_document(html);

    let list_sel =
        Selector::parse(".masonry-text-view-all li").expect("static selector should be valid");
    let link_sel = Selector::parse("a[href]").expect("static selector should be valid");
    let span_sel = Selector::parse("span").expect("static selector should be valid");

    let artists: Vec<WikiArtist> = doc
        .select(&list_sel)
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
            Some(WikiArtist {
                name,
                resource,
                artwork_count,
            })
        })
        .collect();

    if artists.is_empty() {
        // this might be okay for some letters, but log it
        eprintln!("warning: no artists found at {url}");
    }

    Ok(artists)
}

/// Fetch and parse all artists starting with a given letter
pub async fn list_artists_by_letter(
    source: &WikiArtSource,
    client: &Client,
    letter: char,
) -> Result<Vec<WikiArtist>> {
    let url = source.url(&format!("/en/Alphabet/{letter}/text-list"));
    let html = fetch_html(client, &url).await?;
    parse_artists(&html, &url)
}

/// Parse artworks from an artist's text-list page HTML
fn parse_artworks(html: &str, artist: &WikiArtist, url: &str) -> Result<Vec<WikiArtwork>> {
    let doc = Html::parse_document(html);

    let sel = Selector::parse(".painting-list-text-row > a[href]")
        .expect("static selector should be valid");

    let artworks: Vec<WikiArtwork> = doc
        .select(&sel)
        .filter_map(|el| {
            let resource = String::from(el.value().attr("href")?);
            let name = el.inner_html();
            Some(WikiArtwork {
                name,
                resource,
                artist_name: artist.name.clone(),
                artist_resource: artist.resource.clone(),
            })
        })
        .collect();

    if artworks.is_empty() && artist.artwork_count > 0 {
        return Err(Error::UnexpectedHtml {
            context: format!("expected {} artworks but found none", artist.artwork_count),
            url: url.to_string(),
        });
    }

    Ok(artworks)
}

/// Fetch and parse all artworks by an artist
pub async fn list_works_by_artist(
    source: &WikiArtSource,
    client: &Client,
    artist: &WikiArtist,
) -> Result<Vec<WikiArtwork>> {
    let url = format!("{}/all-works/text-list", source.url(&artist.resource));
    let html = fetch_html(client, &url).await?;
    parse_artworks(&html, artist, &url)
}

/// Parse the image URL from an artwork page HTML
fn parse_artwork_image_url(html: &str) -> Result<Option<String>> {
    let doc = Html::parse_document(html);

    let sel = Selector::parse(".wiki-layout-artist-image-wrapper > img[src]")
        .expect("static selector should be valid");

    let restricted =
        Selector::parse(".restricted-msg").expect(".restricted-msg parser should be valid");

    if doc.select(&restricted).next().is_some() {
        return Err(Error::Restricted);
    }

    let Some(src) = doc
        .select(&sel)
        .next()
        .and_then(|el| el.value().attr("src").map(String::from))
    else {
        return Ok(None);
    };

    // strip any image processing suffix after '!'
    Ok(Some(match src.split_once('!') {
        Some((base, _)) => base.to_string(),
        None => src,
    }))
}

/// Stream all artists from WikiArt
fn list_all_artists_stream<'a>(
    source: &'a WikiArtSource,
    client: &'a Client,
) -> impl Stream<Item = Result<WikiArtist>> + 'a {
    stream::iter('a'..='z')
        .map(move |letter| async move { list_artists_by_letter(source, client, letter).await })
        .buffer_unordered(10)
        .flat_map(|result: Result<Vec<WikiArtist>>| match result {
            Ok(artists) => stream::iter(artists.into_iter().map(Ok).collect::<Vec<_>>()),
            Err(e) => {
                eprintln!("error fetching artists: {e}");
                stream::iter(vec![])
            }
        })
}

/// Stream all artworks from WikiArt
fn list_all_artworks_stream<'a>(
    source: &'a WikiArtSource,
    client: &'a Client,
) -> impl Stream<Item = Result<UnifiedArtwork>> + Send + 'a {
    list_all_artists_stream(source, client)
        .filter_map(|r: Result<WikiArtist>| async { r.ok() })
        .map(move |artist: WikiArtist| async move {
            match list_works_by_artist(source, client, &artist).await {
                Ok(works) => works,
                Err(e) => {
                    eprintln!("error fetching works for {}: {e}", artist.name);
                    vec![]
                }
            }
        })
        .buffer_unordered(10)
        .flat_map(stream::iter)
        .map(|artwork| Ok(UnifiedArtwork::from(artwork)))
}
