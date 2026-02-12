use serde::{Deserialize, Serialize};

use super::common::{Creator, Source, UnifiedArtwork};

/// BALAT artwork entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BalatEntry {
    /// IIIF image ID
    pub id: String,
    /// Artwork title
    pub title: String,
    /// Detail page path (e.g., "photo.php?path=...")
    pub detail_path: String,
    /// Creator name
    pub creator_name: Option<String>,
    /// IIIF thumbnail URL
    pub thumbnail_url: String,
}

impl BalatEntry {
    #[allow(dead_code)]
    /// Convert thumbnail URL to full-size image URL
    ///
    /// Thumbnail: `https://iiif.kikirpa.be/iiif/2/[ID]/full/!280,280/0/default.jpg`
    /// Full-size: `https://iiif.kikirpa.be/iiif/2/[ID]/full/max/0/default.jpg`
    pub fn full_image_url(&self) -> String {
        self.thumbnail_url.replace("/full/!280,280/", "/full/max/")
    }

    /// Convert thumbnail URL to full-size image URL
    ///
    /// Thumbnail: `https://iiif.kikirpa.be/iiif/2/[ID]/full/!280,280/0/default.jpg`
    /// Reasonable-size: `https://iiif.kikirpa.be/iiif/2/[ID]/full/!1000,1000/0/default.jpg`
    pub fn reasonable_image_url(&self) -> String {
        self.thumbnail_url
            .replace("/full/!280,280/", "/full/!1000,1000/")
    }
}

impl From<BalatEntry> for UnifiedArtwork {
    fn from(b: BalatEntry) -> Self {
        let image_url = Some(b.reasonable_image_url());
        UnifiedArtwork {
            id: b.id,
            title: b.title,
            resource: b.detail_path,
            source: Source::Balat,
            creator: b.creator_name.map(|name| Creator {
                name,
                resource: None,
            }),
            image_url,
        }
    }
}
