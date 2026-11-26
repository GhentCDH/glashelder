use unstirring::{UnstirClient, UnstirClientConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = UnstirClient::new(UnstirClientConfig::default())?;

    unstirring::inventory::get_buildings(&client).await?;

    Ok(())
}
