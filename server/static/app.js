const clusterListEl = document.getElementById("cluster-list");
const metricTagEl = document.getElementById("metric-tag");
const metricDescriptionEl = document.getElementById("metric-description");
const clusterTitleEl = document.getElementById("cluster-title");
const clusterNameEl = document.getElementById("selected-cluster-name");
const clusterSizeEl = document.getElementById("cluster-size");
const clusterAssetsEl = document.getElementById("cluster-assets");
const assetBreakdownEl = document.getElementById("asset-breakdown");
const clusterSizePill = document.getElementById("cluster-size-pill");
const clusterAssetPill = document.getElementById("cluster-asset-pill");
const modalityStatsEl = document.getElementById("modality-stats");
const itemGridEl = document.getElementById("item-grid");
const refreshButton = document.getElementById("refresh-button");
let clustersCache = [];
let metricInfo = null;
let activeClusterId = null;

const formatSimilarity = (value) =>
    value === null || value === undefined ? "–" : value.toFixed(2);

const renderMetricTag = () => {
    if (!metricInfo) {
        metricTagEl.textContent = "";
        metricDescriptionEl.textContent = "";
        return;
    }
    metricTagEl.textContent = "";
    metricDescriptionEl.textContent = `Transform: ${metricInfo.transform}, metric: ${metricInfo.metric}, scale: ${metricInfo.scale}`;
};

const renderClusterList = () => {
    clusterListEl.innerHTML = "";
    clustersCache.forEach((cluster) => {
        const li = document.createElement("li");
        li.className = "cluster-entry";
        li.dataset.clusterId = cluster.cluster_id;
        const fused = cluster.modality_summary?.fused?.average_similarity;
        li.innerHTML = `
      <strong>Cluster ${cluster.cluster_id}</strong>
      <span>${cluster.size} entities · ${cluster.unique_assets} assets</span>
      <span>Fused avg: ${formatSimilarity(fused)}</span>
    `;
        li.onclick = () => loadCluster(cluster.cluster_id);
        clusterListEl.appendChild(li);
    });
    highlightActiveCluster();
};

const highlightActiveCluster = () => {
    document.querySelectorAll(".cluster-entry").forEach((entry) => {
        entry.classList.toggle(
            "active",
            Number(entry.dataset.clusterId) === activeClusterId,
        );
    });
};

const renderClusterSummary = (summary) => {
    if (!summary) return;
    clusterTitleEl.textContent = `Cluster ${summary.cluster_id}`;
    clusterNameEl.textContent = "Cluster highlights";
    clusterSizeEl.textContent = summary.size;
    clusterAssetsEl.textContent = summary.unique_assets;
    const breakdown = Object.entries(summary.asset_breakdown || {})
        .map(([type, count]) => `${type}: ${count}`)
        .join(" · ");
    assetBreakdownEl.textContent = breakdown || "–";
    clusterSizePill.textContent = `Entities • ${summary.size}`;
    clusterAssetPill.textContent = `Assets • ${summary.unique_assets}`;
};

const renderModalityStats = (summary) => {
    modalityStatsEl.innerHTML = "";
    if (!summary?.modality_summary) {
        modalityStatsEl.innerHTML =
            "<p class='modal-empty'>No modalities available.</p>";
        return;
    }
    Object.entries(summary.modality_summary).forEach(([modality, data]) => {
        const stat = document.createElement("article");
        stat.className = "modality-stat";
        const avg = data.average_similarity ?? 0;
        const pct = Math.min(100, avg * 100);
        stat.innerHTML = `
      <strong>${modality}</strong>
      <span>${data.coverage}/${summary.size} entities</span>
      <div class="bar"><span style="width:${pct}%"></span></div>
      <p>avg sim: ${formatSimilarity(data.average_similarity)}</p>
    `;
        modalityStatsEl.appendChild(stat);
    });
};

const createMediaElement = (item) => {
    const wrapper = document.createElement("div");
    wrapper.className = "media-wrapper";
    if (item.asset_type === "video") {
        const video = document.createElement("video");
        video.src = `/media/${item.asset_id}`;
        video.controls = true;
        video.muted = true;
        video.loop = true;
        video.autoplay = true;
        video.playsInline = true;
        wrapper.appendChild(video);
    } else {
        const img = document.createElement("img");
        img.src = `/media/${item.asset_id}`;
        img.alt = item.asset_basename;
        wrapper.appendChild(img);
    }
    return wrapper;
};

const renderItems = (items = []) => {
    itemGridEl.innerHTML = "";
    if (!items.length) {
        itemGridEl.innerHTML =
            "<p class='placeholder'>No assets were found for this cluster.</p>";
        return;
    }
    items.forEach((item) => {
        const card = document.createElement("article");
        card.className = "entity-card";
        const media = createMediaElement(item);
        const info = document.createElement("div");
        info.className = "card-info";
        const badges = document.createElement("div");
        badges.className = "badges";
        Object.entries(item.similarities).forEach(([modality, value]) => {
            const badge = document.createElement("span");
            badge.className = "mod-badge";
            badge.innerHTML = `<strong>${modality}</strong><span>${formatSimilarity(
                value,
            )}</span>`;
            badges.appendChild(badge);
        });
        info.innerHTML = `
      <p class="card-title">Entity ${item.entity_id}</p>
      <p class="card-subtitle">${item.asset_basename} · ${item.asset_type}</p>
      <div class="card-meta">
        <span>${item.frame_count} frames</span>
        <span>${item.faces_total} faces</span>
      </div>
    `;
        info.appendChild(badges);
        card.appendChild(media);
        card.appendChild(info);
        card.onclick = () => window.open(`/media/${item.asset_id}`, "_blank");
        itemGridEl.appendChild(card);
    });
};

const loadCluster = async (clusterId) => {
    activeClusterId = clusterId;
    highlightActiveCluster();
    const res = await fetch(`/api/cluster/${clusterId}`);
    if (!res.ok) {
        return;
    }
    const data = await res.json();
    renderClusterSummary(data.summary);
    renderModalityStats(data.summary);
    renderItems(data.items);
};

const loadClusters = async () => {
    const res = await fetch("/api/clusters");
    if (!res.ok) {
        return;
    }
    const data = await res.json();
    clustersCache = data.clusters;
    metricInfo = data.similarity_metric;
    renderMetricTag();
    renderClusterList();
};

const refreshClusters = async () => {
    const res = await fetch("/api/refresh");
    if (!res.ok) {
        return;
    }
    clustersCache = [];
    activeClusterId = null;
    itemGridEl.innerHTML = "<p class='placeholder'>Refreshing clusters…</p>";
    await loadClusters();
    clusterTitleEl.textContent = "Choose a cluster to inspect";
    clusterNameEl.textContent = "Select a cluster";
    clusterSizeEl.textContent = "–";
    clusterAssetsEl.textContent = "–";
    assetBreakdownEl.textContent = "–";
    modalityStatsEl.innerHTML = "";
};

refreshButton.addEventListener("click", refreshClusters);
loadClusters();

const params = new URLSearchParams(window.location.search);
const initialCluster = params.get("cluster_id");
if (initialCluster) {
    loadCluster(Number(initialCluster));
}
