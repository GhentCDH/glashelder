pub mod balat;
pub mod wikiart;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;

use crate::error::Result;
use crate::model::UnifiedArtwork;

/// Trait for artwork data sources
#[async_trait]
#[allow(dead_code)]
pub trait ArtSource: Send + Sync {
    /// Human-readable name of the source
    fn name(&self) -> &'static str;

    /// Base URL for this source
    fn base_url(&self) -> &str;

    /// List all artworks from this source as a stream
    fn list_artworks<'a>(
        &'a self,
        client: &'a Client,
    ) -> Box<dyn Stream<Item = Result<UnifiedArtwork>> + Send + Unpin + 'a>;

    /// Get the image URL for an artwork (may require additional fetch)
    async fn get_image_url(
        &self,
        client: &Client,
        artwork: &UnifiedArtwork,
    ) -> Result<Option<String>>;
}

pub use balat::BalatSource;
pub use wikiart::WikiArtSource;
