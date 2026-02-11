use reqwest::Client;
use scraper::{Html, Selector};

use crate::client::{fetch_html, wiki_url};
use crate::error::{Error, Result};
use crate::model::{Artist, Artwork};

/// Parse artists from the alphabet listing page HTML
fn parse_artists(html: &str, url: &str) -> Result<Vec<Artist>> {
    let doc = Html::parse_document(html);

    let list_sel =
        Selector::parse(".masonry-text-view-all li").expect("static selector should be valid");
    let link_sel = Selector::parse("a[href]").expect("static selector should be valid");
    let span_sel = Selector::parse("span").expect("static selector should be valid");

    let artists: Vec<Artist> = doc
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
            Some(Artist {
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
pub async fn list_artists_by_letter(client: &Client, letter: char) -> Result<Vec<Artist>> {
    let url = wiki_url(&format!("/en/Alphabet/{letter}/text-list"));
    let html = fetch_html(client, &url).await?;
    parse_artists(&html, &url)
}

/// Parse artworks from an artist's text-list page HTML
fn parse_artworks(html: &str, artist: &Artist, url: &str) -> Result<Vec<Artwork>> {
    let doc = Html::parse_document(html);

    let sel = Selector::parse(".painting-list-text-row > a[href]")
        .expect("static selector should be valid");

    let artworks: Vec<Artwork> = doc
        .select(&sel)
        .filter_map(|el| {
            let resource = String::from(el.value().attr("href")?);
            let name = el.inner_html();
            Some(Artwork {
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
pub async fn list_works_by_artist(client: &Client, artist: &Artist) -> Result<Vec<Artwork>> {
    let url = format!("{}/all-works/text-list", wiki_url(&artist.resource));
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

/// Fetch the high-res image URL for an artwork
pub async fn get_artwork_image_url(client: &Client, artwork: &Artwork) -> Result<Option<String>> {
    let url = wiki_url(&artwork.resource);
    let html = fetch_html(client, &url).await?;
    parse_artwork_image_url(&html)
}
