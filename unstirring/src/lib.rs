use std::{
    num::NonZeroU32,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use governor::{DefaultKeyedRateLimiter, Quota};
use reqwest::{
    Client, ClientBuilder, RequestBuilder, Response,
    header::{ACCEPT, HeaderMap, HeaderName, HeaderValue, USER_AGENT as UA},
};
use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::Semaphore;

pub mod image;
pub mod inventory;

const USER_AGENT: &str = "Ghent Centre for Digital Humanities <sysadmin.ghentcdh@ugent.be>";

static REQUESTS: AtomicU64 = AtomicU64::new(0);
static START: LazyLock<Instant> = LazyLock::new(Instant::now);

pub struct UnstirClientConfig {
    concurrency: usize,
    per_second: NonZeroU32,
}

impl Default for UnstirClientConfig {
    fn default() -> Self {
        Self {
            concurrency: 2,
            per_second: NonZeroU32::new(6).expect("6 is nonzero"),
        }
    }
}

#[must_use]
pub struct UnstirRequest<'a> {
    client: &'a UnstirClient,
    inner: RequestBuilder,
}

impl<'a> UnstirRequest<'a> {
    pub fn query<T: Serialize + ?Sized>(self, query: &T) -> Self {
        Self {
            inner: self.inner.query(query),
            ..self
        }
    }

    pub fn header<K, V>(self, key: K, value: V) -> Self
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        Self {
            inner: self.inner.header(key, value),
            ..self
        }
    }

    pub async fn json<T: DeserializeOwned>(self) -> anyhow::Result<T> {
        let client = self.client;
        client
            .limited(self.inner, |r| async { Ok(r.json().await?) })
            .await
    }

    pub async fn bytes(self) -> anyhow::Result<bytes::Bytes> {
        let client = self.client;
        client
            .limited(self.inner, |r| async { Ok(r.bytes().await?) })
            .await
    }
}

/// Reqwest-backed managed http client with rate limiting and concurrency limits.
///
/// Automatically sets `Accept: application/json` and `User-Agent` headings
#[derive(Clone)]
pub struct UnstirClient {
    client: Client,
    permits: Arc<Semaphore>,
    rate: Arc<DefaultKeyedRateLimiter<String>>,
}

impl<'a> UnstirClient {
    pub fn new(config: UnstirClientConfig) -> anyhow::Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(UA, HeaderValue::from_static(USER_AGENT));

        let client = ClientBuilder::new().default_headers(headers).build()?;

        Ok(Self {
            client,
            permits: Arc::new(Semaphore::new(config.concurrency)),
            rate: Arc::new(DefaultKeyedRateLimiter::keyed(Quota::per_second(
                config.per_second,
            ))),
        })
    }

    pub fn get(&'a self, url: impl reqwest::IntoUrl) -> UnstirRequest<'a> {
        UnstirRequest {
            client: self,
            inner: self.client.get(url),
        }
    }

    async fn limited<T, F, Fut>(&self, req: RequestBuilder, read: F) -> anyhow::Result<T>
    where
        F: FnOnce(Response) -> Fut,
        Fut: Future<Output = anyhow::Result<T>>,
    {
        let req = req.build()?;
        let host = req.url().host_str().unwrap_or_default().to_owned();
        let _permit = self.permits.acquire().await?;
        self.rate.until_key_ready(&host).await;
        let count = REQUESTS.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "starting request {count} at {:.3?} ({host})",
            START.elapsed()
        );
        let result = read(self.client.execute(req).await?.error_for_status()?).await;
        eprintln!(
            "\tfinished request {count} at {:.3?} ({host})",
            START.elapsed()
        );
        result
    }
}
