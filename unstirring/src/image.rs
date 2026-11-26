use std::path::Path;

use bulb_dither::{diffusion::Kernel::FloydSteinberg, quantize};
use image::DynamicImage;

const IMAGE_PRINT_WIDTH: u32 = 20;
const IMAGE_WIDTH: u32 = 100;

pub fn save_image(bytes: &[u8], path: impl AsRef<Path>) -> anyhow::Result<()> {
    let img = image::load_from_memory(bytes)?;
    img.save(path).map_err(Into::into)
}

pub fn print_image_bytes(bytes: &[u8]) -> anyhow::Result<DynamicImage> {
    let img = image::load_from_memory(bytes)?;

    let width = img.width();
    let height = img.height();

    let new_height = {
        let factor = width as f32 / (IMAGE_WIDTH as f32);
        (height as f32 / factor) as u32
    };

    let mut img = img
        .resize(
            IMAGE_WIDTH,
            new_height,
            image::imageops::FilterType::Triangle,
        )
        .into_rgba8();

    let palette = bulb_dither::palette::extract(
        &img,
        &bulb_dither::palette::ExtractOptions {
            colours: 16,
            hull_weight: 5000.,
            fast: true,
        },
    );
    FloydSteinberg.dither(&mut img, &quantize::PaletteLut::new(&palette), true);

    let img = Into::<DynamicImage>::into(img).resize(
        IMAGE_WIDTH * 3,
        new_height * 3,
        image::imageops::FilterType::Nearest,
    );

    let config = viuer::Config {
        width: Some(IMAGE_PRINT_WIDTH),
        absolute_offset: false,
        ..Default::default()
    };
    viuer::print(&img, &config)?;
    Ok(img)
}
