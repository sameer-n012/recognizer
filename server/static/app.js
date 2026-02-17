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
const tabButtons = document.querySelectorAll(".tab-button");
const dashboardView = document.getElementById("dashboard-view");
const editView = document.getElementById("edit-view");
const editClusterTitleEl = document.getElementById("edit-cluster-title");
const editClusterSizePill = document.getElementById("edit-cluster-size-pill");
const editClusterTotalPill = document.getElementById("edit-cluster-total-pill");
const editItemGridEl = document.getElementById("edit-item-grid");
const suggestedClustersEl = document.getElementById("suggested-clusters");
const manualClusterInput = document.getElementById("manual-cluster-id");
const moveManualButton = document.getElementById("move-to-manual");
const mergeManualButton = document.getElementById("merge-with-manual");
const splitSelectionButton = document.getElementById("split-selection");

let clustersCache = [];
let metricInfo = null;
let activeClusterId = null;
let activeTab = "dashboard";
let selectedEntityIds = new Set();

const formatSimilarity = (value) =>
    value === null || value === undefined ? "–" : value.toFixed(2);

const renderMetricTag = () => {
    if (!metricInfo) {
        metricTagEl.textContent = "Similarity metric unknown";
        metricDescriptionEl.textContent = "";
        return;
    }
    metricTagEl.textContent = `${metricInfo.transform.toUpperCase()}(${metricInfo.metric})`;
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

const setActiveTab = (tab) => {
    activeTab = tab;
    tabButtons.forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.tab === tab);
    });
    dashboardView.classList.toggle("view-hidden", tab !== "dashboard");
    editView.classList.toggle("view-hidden", tab !== "edit");
};

const renderEditItems = (items = []) => {
    editItemGridEl.innerHTML = "";
    if (!items.length) {
        editItemGridEl.innerHTML =
            "<p class='placeholder'>No assets were found for this cluster.</p>";
        return;
    }
    items.forEach((item) => {
        const card = document.createElement("article");
        card.className = "entity-card selectable";
        card.dataset.entityId = item.entity_id;
        const media = createMediaElement(item);
        const info = document.createElement("div");
        info.className = "card-info";
        info.innerHTML = `
      <p class="card-title">Entity ${item.entity_id}</p>
      <p class="card-subtitle">${item.asset_basename} · ${item.asset_type}</p>
      <div class="card-meta">
        <span>${item.frame_count} frames</span>
        <span>${item.faces_total} faces</span>
      </div>
    `;
        const toggle = document.createElement("button");
        toggle.className = "select-toggle";
        toggle.innerHTML = "<span>○</span><span>Select</span>";
        toggle.onclick = (event) => {
            event.stopPropagation();
            toggleSelection(item.entity_id, card, toggle);
        };
        card.appendChild(toggle);
        card.appendChild(media);
        card.appendChild(info);
        card.onclick = () => window.open(`/media/${item.asset_id}`, "_blank");
        editItemGridEl.appendChild(card);
    });
    syncSelectionState();
};

const toggleSelection = (entityId, card, toggle) => {
    if (selectedEntityIds.has(entityId)) {
        selectedEntityIds.delete(entityId);
    } else {
        selectedEntityIds.add(entityId);
    }
    card.classList.toggle("selected", selectedEntityIds.has(entityId));
    toggle.innerHTML = selectedEntityIds.has(entityId)
        ? "<span>●</span><span>Selected</span>"
        : "<span>○</span><span>Select</span>";
    updateSelectionPills();
};

const syncSelectionState = () => {
    document.querySelectorAll(".entity-card.selectable").forEach((card) => {
        const entityId = card.dataset.entityId;
        const toggle = card.querySelector(".select-toggle");
        const selected = selectedEntityIds.has(entityId);
        card.classList.toggle("selected", selected);
        if (toggle) {
            toggle.innerHTML = selected
                ? "<span>●</span><span>Selected</span>"
                : "<span>○</span><span>Select</span>";
        }
    });
    updateSelectionPills();
};

const updateSelectionPills = () => {
    editClusterSizePill.textContent = `Selected • ${selectedEntityIds.size}`;
    const disabled = selectedEntityIds.size === 0;
    moveManualButton.disabled = disabled;
    mergeManualButton.disabled = disabled;
    splitSelectionButton.disabled = disabled;
};

const renderSuggestedClusters = (suggestions = []) => {
    suggestedClustersEl.innerHTML = "";
    if (!suggestions.length) {
        suggestedClustersEl.innerHTML =
            "<p class='placeholder'>No suggestions available.</p>";
        return;
    }
    suggestions.forEach((item) => {
        const row = document.createElement("div");
        row.className = "suggested-item";
        row.innerHTML = `
      <strong>Cluster ${item.cluster_id}</strong>
      <span>Similarity ${formatSimilarity(item.similarity)}</span>
    `;
        const moveButton = document.createElement("button");
        moveButton.className = "action-btn";
        moveButton.textContent = "Move selected";
        moveButton.disabled = selectedEntityIds.size === 0;
        moveButton.onclick = () => moveSelectedEntities(item.cluster_id);
        const mergeButton = document.createElement("button");
        mergeButton.className = "action-btn secondary";
        mergeButton.textContent = "Merge cluster";
        mergeButton.onclick = () => mergeClusters(item.cluster_id);
        row.appendChild(moveButton);
        row.appendChild(mergeButton);
        suggestedClustersEl.appendChild(row);
    });
};

const loadSuggestions = async (clusterId) => {
    const res = await fetch(`/api/cluster/${clusterId}/suggestions?limit=8`);
    if (!res.ok) {
        return;
    }
    const data = await res.json();
    renderSuggestedClusters(data.clusters || []);
};

const postEditAction = async (payload) => {
    const res = await fetch("/api/cluster/edit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (!res.ok) {
        return null;
    }
    return res.json();
};

const moveSelectedEntities = async (targetClusterId) => {
    if (!selectedEntityIds.size || targetClusterId === null) return;
    await postEditAction({
        action: "move_entities",
        entity_ids: Array.from(selectedEntityIds),
        target_cluster_id: Number(targetClusterId),
    });
    selectedEntityIds.clear();
    await loadClusters();
    if (activeClusterId !== null) {
        await loadCluster(activeClusterId);
    }
};

const mergeClusters = async (targetClusterId) => {
    if (activeClusterId === null || targetClusterId === null) return;
    await postEditAction({
        action: "merge_clusters",
        source_cluster_id: Number(activeClusterId),
        target_cluster_id: Number(targetClusterId),
    });
    selectedEntityIds.clear();
    await loadClusters();
    await loadCluster(targetClusterId);
};

const splitSelectedEntities = async () => {
    if (!selectedEntityIds.size || activeClusterId === null) return;
    const response = await postEditAction({
        action: "split_cluster",
        source_cluster_id: Number(activeClusterId),
        entity_ids: Array.from(selectedEntityIds),
    });
    selectedEntityIds.clear();
    await loadClusters();
    if (response?.details?.new_cluster_id !== undefined) {
        await loadCluster(response.details.new_cluster_id);
    } else if (activeClusterId !== null) {
        await loadCluster(activeClusterId);
    }
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
    editClusterTitleEl.textContent = `Editing Cluster ${data.summary.cluster_id}`;
    editClusterTotalPill.textContent = `Entities • ${data.summary.size}`;
    renderEditItems(data.items);
    await loadSuggestions(clusterId);
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
tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
        setActiveTab(button.dataset.tab);
    });
});
moveManualButton.addEventListener("click", () => {
    const target = manualClusterInput.value;
    if (!target) return;
    moveSelectedEntities(Number(target));
});
mergeManualButton.addEventListener("click", () => {
    const target = manualClusterInput.value;
    if (!target) return;
    mergeClusters(Number(target));
});
splitSelectionButton.addEventListener("click", splitSelectedEntities);

loadClusters();
setActiveTab("dashboard");
