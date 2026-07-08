const graphCanvas = document.getElementById("graph-canvas");
const selectionCountEl = document.getElementById("selection-count");
const assetSelectionCountEl = document.getElementById("asset-selection-count");
const prevNodeBtn = document.getElementById("prev-node");
const nextNodeBtn = document.getElementById("next-node");
const undoBtn = document.getElementById("undo-edit");
const redoBtn = document.getElementById("redo-edit");
const clearSelectionBtn = document.getElementById("clear-selection");
const zoomFitBtn = document.getElementById("zoom-fit");
const clusterSearch = document.getElementById("cluster-search");
const openEditorBtn = document.getElementById("open-editor");
const graphPanelTitle = document.getElementById("graph-panel-title");
const graphPanelSubtitle = document.getElementById("graph-panel-subtitle");
const graphEntitiesEl = document.getElementById("graph-entities");
const graphSplitBtn = document.getElementById("graph-split");
const graphClusterSizeEl = document.getElementById("graph-cluster-size");
const graphClusterAssetsEl = document.getElementById("graph-cluster-assets");
const graphAssetBreakdownEl = document.getElementById("graph-asset-breakdown");
const graphMetricDescriptionEl = document.getElementById(
    "graph-metric-description",
);
const graphModalityStatsEl = document.getElementById("graph-modality-stats");
const contextMenu = document.getElementById("context-menu");
const actionModal = document.getElementById("action-modal");
const modalTitle = document.getElementById("modal-title");
const modalDescription = document.getElementById("modal-description");
const modalConfirm = document.getElementById("modal-confirm");
const modalCancel = document.getElementById("modal-cancel");
const hoverTooltip = document.getElementById("hover-tooltip");
const tooltipTitle = document.getElementById("tooltip-title");
const tooltipImage = document.getElementById("tooltip-image");
const tooltipVideo = document.getElementById("tooltip-video");

let nodes = [];
let clustersMap = new Map();
let nodeElements = new Map();
let selectedClusterIds = new Set();
let selectedEntityIds = new Set();
let assetSelectionMap = new Map();
let metricInfo = null;
let activeClusterId = null;
let pendingAction = null;
let contextTargetId = null;
let hoverTimer = null;
let hoverClusterId = null;

const SVG_NS = "http://www.w3.org/2000/svg";
const viewBox = { x: 0, y: 0, w: 1000, h: 1000 };

const svg = document.createElementNS(SVG_NS, "svg");
svg.setAttribute("width", "100%");
svg.setAttribute("height", "100%");
svg.setAttribute(
    "viewBox",
    `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`,
);
svg.style.cursor = "grab";

graphCanvas.appendChild(svg);

const overlayGroup = document.createElementNS(SVG_NS, "g");
const nodesGroup = document.createElementNS(SVG_NS, "g");
svg.appendChild(overlayGroup);
svg.appendChild(nodesGroup);

const formatSimilarity = (value) =>
    value === null || value === undefined ? "–" : value.toFixed(2);

const updateViewBox = () => {
    svg.setAttribute(
        "viewBox",
        `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`,
    );
};

const toSvgPoint = (clientX, clientY) => {
    const rect = svg.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * viewBox.w + viewBox.x;
    const y = ((clientY - rect.top) / rect.height) * viewBox.h + viewBox.y;
    return { x, y };
};

const renderNodes = () => {
    nodesGroup.innerHTML = "";
    nodeElements.clear();
    [...nodes]
        .sort((a, b) => (b.size || 0) - (a.size || 0))
        .forEach((node) => {
            const circle = document.createElementNS(SVG_NS, "circle");
            const radius = Math.max(6, Math.sqrt(node.size) * 2.8);
            circle.setAttribute("cx", node.x * 1000);
            circle.setAttribute("cy", node.y * 1000);
            circle.setAttribute("r", radius);
            circle.setAttribute("fill", "rgba(100, 221, 255, 0.7)");
            circle.setAttribute("stroke", "rgba(255,255,255,0.4)");
            circle.setAttribute("stroke-width", "1");
            circle.dataset.clusterId = node.cluster_id;
            circle.style.cursor = "pointer";

            circle.addEventListener("click", (event) => {
                event.stopPropagation();
                handleNodeClick(node.cluster_id, event);
            });
            circle.addEventListener("mouseenter", (event) => {
                startHoverPreview(
                    node.cluster_id,
                    event.clientX,
                    event.clientY,
                );
            });
            circle.addEventListener("mouseleave", () => {
                cancelHoverPreview();
            });
            circle.addEventListener("mousemove", (event) => {
                if (hoverClusterId === node.cluster_id) {
                    positionTooltip(event.clientX, event.clientY);
                }
            });
            circle.addEventListener("contextmenu", (event) => {
                event.preventDefault();
                event.stopPropagation();
                openContextMenu(event.clientX, event.clientY, node.cluster_id);
            });
            nodesGroup.appendChild(circle);
            nodeElements.set(node.cluster_id, circle);
        });
    updateSelectionStyles();
};

const updateSelectionStyles = () => {
    nodeElements.forEach((circle, clusterId) => {
        const selected = selectedClusterIds.has(clusterId);
        circle.setAttribute(
            "fill",
            selected ? "rgba(249, 115, 22, 0.85)" : "rgba(100, 221, 255, 0.7)",
        );
        circle.setAttribute("stroke-width", selected ? "2" : "1");
    });
    selectionCountEl.textContent = `${selectedClusterIds.size} clusters selected`;
    assetSelectionCountEl.textContent = `${assetSelectionMap.size} assets selected`;
    updateActionState();
};

const updateActionState = () => {
    const singleCluster = selectedClusterIds.size === 1;
    const hasAssets = assetSelectionMap.size > 0;
    graphSplitBtn.disabled = !(singleCluster && hasAssets);
};

const handleNodeClick = (clusterId, event) => {
    if (event.metaKey || event.ctrlKey) {
        if (selectedClusterIds.has(clusterId)) {
            selectedClusterIds.delete(clusterId);
        } else {
            selectedClusterIds.add(clusterId);
        }
    } else {
        selectedClusterIds = new Set([clusterId]);
    }
    activeClusterId = clusterId;
    updateSelectionStyles();
    if (selectedClusterIds.size === 1) {
        loadClusterDetails(clusterId);
    } else {
        activeClusterId = null;
        selectedEntityIds.clear();
        assetSelectionMap.clear();
        updateSelectionStyles();
        renderMultiClusterSummary();
    }
};

const selectSingleCluster = (clusterId) => {
    selectedClusterIds = new Set([clusterId]);
    activeClusterId = clusterId;
    selectedEntityIds.clear();
    assetSelectionMap.clear();
    updateSelectionStyles();
    loadClusterDetails(clusterId);
};

const stepClusterSelection = (direction) => {
    if (!nodes.length) return;
    const sorted = [...nodes].sort((a, b) => a.cluster_id - b.cluster_id);
    const currentId = activeClusterId ?? sorted[0].cluster_id;
    const currentIndex = sorted.findIndex(
        (node) => node.cluster_id === currentId,
    );
    const nextIndex =
        (currentIndex + direction + sorted.length) % sorted.length;
    selectSingleCluster(sorted[nextIndex].cluster_id);
};

const clearSelection = () => {
    selectedClusterIds.clear();
    selectedEntityIds.clear();
    assetSelectionMap.clear();
    activeClusterId = null;
    updateSelectionStyles();
    graphPanelTitle.textContent = "No cluster selected";
    graphPanelSubtitle.textContent = "Select a node to inspect.";
    graphEntitiesEl.innerHTML = "";
    graphClusterSizeEl.textContent = "–";
    graphClusterAssetsEl.textContent = "–";
    graphAssetBreakdownEl.textContent = "–";
    graphMetricDescriptionEl.textContent = "–";
    graphModalityStatsEl.innerHTML = "";
};

clearSelectionBtn.addEventListener("click", clearSelection);

const zoomFit = () => {
    if (!nodes.length) {
        viewBox.x = 0;
        viewBox.y = 0;
        viewBox.w = 1000;
        viewBox.h = 1000;
        updateViewBox();
        return;
    }
    const xs = nodes.map((node) => node.x * 1000);
    const ys = nodes.map((node) => node.y * 1000);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padding = 120;
    viewBox.x = minX - padding;
    viewBox.y = minY - padding;
    viewBox.w = maxX - minX + padding * 2;
    viewBox.h = maxY - minY + padding * 2;
    updateViewBox();
};

zoomFitBtn.addEventListener("click", zoomFit);

const loadGraph = async (preserveView = false) => {
    const [graphRes, clusterRes] = await Promise.all([
        fetch("/api/cluster_graph"),
        fetch("/api/clusters"),
    ]);
    if (!graphRes.ok || !clusterRes.ok) return;
    const data = await graphRes.json();
    const clusterData = await clusterRes.json();
    metricInfo = clusterData.similarity_metric;
    clustersMap = new Map(
        (clusterData.clusters || []).map((cluster) => [
            cluster.cluster_id,
            cluster,
        ]),
    );
    nodes = data.nodes || [];
    renderNodes();
    if (!preserveView) {
        zoomFit();
    }
};

const loadClusterDetails = async (clusterId) => {
    const res = await fetch(`/api/cluster/${clusterId}`);
    if (!res.ok) return;
    const data = await res.json();
    graphPanelTitle.textContent = `Cluster ${data.summary.cluster_id}`;
    graphPanelSubtitle.textContent = `${data.summary.size} entities · ${data.summary.unique_assets} assets`;
    graphClusterSizeEl.textContent = data.summary.size;
    graphClusterAssetsEl.textContent = data.summary.unique_assets;
    const breakdown = Object.entries(data.summary.asset_breakdown || {})
        .map(([type, count]) => `${type}: ${count}`)
        .join(" · ");
    graphAssetBreakdownEl.textContent = breakdown || "–";
    if (metricInfo) {
        graphMetricDescriptionEl.textContent = `Transform: ${metricInfo.transform}, metric: ${metricInfo.metric}, scale: ${metricInfo.scale}`;
    }
    renderModalityStats(data.summary);
    renderEntities(data.items || []);
};

const renderMultiClusterSummary = () => {
    const selectedClusters = Array.from(selectedClusterIds)
        .map((cid) => clustersMap.get(cid))
        .filter(Boolean);
    if (!selectedClusters.length) {
        return;
    }
    const totalEntities = selectedClusters.reduce(
        (sum, cluster) => sum + (cluster.size || 0),
        0,
    );
    const totalAssets = selectedClusters.reduce(
        (sum, cluster) => sum + (cluster.unique_assets || 0),
        0,
    );
    const breakdown = selectedClusters
        .flatMap((cluster) => Object.entries(cluster.asset_breakdown || {}))
        .reduce((acc, [type, count]) => {
            acc[type] = (acc[type] || 0) + count;
            return acc;
        }, {});

    graphPanelTitle.textContent = `${selectedClusters.length} clusters selected`;
    graphPanelSubtitle.textContent = `${totalEntities} entities · ${totalAssets} assets`;
    graphClusterSizeEl.textContent = totalEntities;
    graphClusterAssetsEl.textContent = totalAssets;
    graphAssetBreakdownEl.textContent =
        Object.entries(breakdown)
            .map(([type, count]) => `${type}: ${count}`)
            .join(" · ") || "–";
    if (metricInfo) {
        graphMetricDescriptionEl.textContent = `Transform: ${metricInfo.transform}, metric: ${metricInfo.metric}, scale: ${metricInfo.scale}`;
    }
    renderModalityStats({
        size: totalEntities,
        modality_summary: selectedClusters.reduce((acc, cluster) => {
            Object.entries(cluster.modality_summary || {}).forEach(
                ([modality, data]) => {
                    if (!acc[modality]) {
                        acc[modality] = { total: 0, count: 0, coverage: 0 };
                    }
                    if (data.average_similarity !== null) {
                        acc[modality].total +=
                            (data.average_similarity || 0) *
                            (data.coverage || 0);
                        acc[modality].coverage += data.coverage || 0;
                    }
                    acc[modality].count += 1;
                },
            );
            return acc;
        }, {}),
    });
    graphEntitiesEl.innerHTML =
        "<p class='placeholder'>Select a single cluster to view assets.</p>";
};
const renderModalityStats = (summary) => {
    graphModalityStatsEl.innerHTML = "";
    if (!summary?.modality_summary) {
        graphModalityStatsEl.innerHTML =
            "<p class='modal-empty'>No modalities available.</p>";
        return;
    }
    Object.entries(summary.modality_summary).forEach(([modality, data]) => {
        const average =
            data.average_similarity ??
            (data.coverage
                ? (data.total || 0) / Math.max(1, data.coverage)
                : null);
        const stat = document.createElement("article");
        stat.className = "modality-stat";
        const avg = average ?? 0;
        const pct = Math.min(100, avg * 100);
        stat.innerHTML = `
      <strong>${modality}</strong>
      <span>${data.coverage || 0}/${summary.size} entities</span>
      <div class=\"bar\"><span style=\"width:${pct}%\"></span></div>
      <p>avg sim: ${formatSimilarity(average)}</p>
    `;
        graphModalityStatsEl.appendChild(stat);
    });
};

const renderEntities = (items) => {
    graphEntitiesEl.innerHTML = "";
    selectedEntityIds.clear();
    assetSelectionMap.clear();
    if (!items.length) {
        graphEntitiesEl.innerHTML =
            "<p class='placeholder'>No entities available.</p>";
        return;
    }
    if (selectedClusterIds.size !== 1) {
        graphEntitiesEl.innerHTML =
            "<p class='placeholder'>Select a single cluster to view assets.</p>";
        updateSelectionStyles();
        return;
    }
    const byAsset = new Map();
    items.forEach((item) => {
        if (!byAsset.has(item.asset_id)) {
            byAsset.set(item.asset_id, { ...item, entity_ids: [] });
        }
        byAsset.get(item.asset_id).entity_ids.push(item.entity_id);
    });

    Array.from(byAsset.values()).forEach((item) => {
        const card = document.createElement("div");
        card.className = "graph-entity-card";
        card.dataset.assetId = item.asset_id;

        const media = document.createElement(
            item.asset_type === "video" ? "video" : "img",
        );
        media.src = `/media/${item.asset_id}`;
        if (item.asset_type === "video") {
            media.muted = true;
            media.loop = true;
            media.playsInline = true;
        } else {
            media.alt = item.asset_basename;
        }
        media.addEventListener("click", () => {
            window.open(`/media/${item.asset_id}`, "_blank");
        });

        const meta = document.createElement("div");
        meta.className = "graph-entity-meta";
        const toggle = document.createElement("button");
        toggle.textContent = "Select";
        toggle.addEventListener("click", () => {
            if (assetSelectionMap.has(item.asset_id)) {
                assetSelectionMap.delete(item.asset_id);
                item.entity_ids.forEach((entityId) => {
                    selectedEntityIds.delete(entityId);
                });
                toggle.textContent = "Select";
                card.classList.remove("selected");
            } else {
                assetSelectionMap.set(item.asset_id, item.entity_ids);
                item.entity_ids.forEach((entityId) => {
                    selectedEntityIds.add(entityId);
                });
                toggle.textContent = "Selected";
                card.classList.add("selected");
            }
            updateSelectionStyles();
        });
        meta.appendChild(toggle);
        card.appendChild(media);
        card.appendChild(meta);
        graphEntitiesEl.appendChild(card);
    });
    updateSelectionStyles();
};

const postEditAction = async (payload) => {
    const res = await fetch("/api/cluster/edit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (!res.ok) return null;
    return res.json();
};

const mergeSelectedClusters = async (targetClusterId) => {
    const clusterIds = Array.from(selectedClusterIds);
    if (clusterIds.length < 2) {
        return;
    }
    const response = await postEditAction({
        action: "merge_selected",
        cluster_ids: clusterIds,
    });
    const newClusterId = response?.details?.new_cluster_id;
    await loadGraph(true);
    if (newClusterId !== undefined) {
        selectedClusterIds = new Set([newClusterId]);
        activeClusterId = newClusterId;
        updateSelectionStyles();
        animateNode(newClusterId, "node-merge");
        await loadClusterDetails(newClusterId);
    } else {
        clearSelection();
    }
};

const moveSelectedEntities = async (targetClusterId) => {
    if (!selectedEntityIds.size) return;
    await postEditAction({
        action: "move_entities",
        entity_ids: Array.from(selectedEntityIds),
        target_cluster_id: Number(targetClusterId),
    });
    selectedEntityIds.clear();
    await loadGraph(true);
    if (activeClusterId !== null) {
        selectedClusterIds = new Set([activeClusterId]);
        updateSelectionStyles();
        await loadClusterDetails(activeClusterId);
        animateNode(activeClusterId, "node-move");
    }
    if (targetClusterId !== null) {
        animateNode(Number(targetClusterId), "node-move");
    }
};

const runUndo = async () => {
    const res = await fetch("/api/cluster/undo", { method: "POST" });
    if (!res.ok) return;
    await loadGraph(true);
    clearSelection();
};

const runRedo = async () => {
    const res = await fetch("/api/cluster/redo", { method: "POST" });
    if (!res.ok) return;
    await loadGraph(true);
    clearSelection();
};

const splitSelectedEntities = async () => {
    if (!selectedEntityIds.size || activeClusterId === null) return;
    const response = await postEditAction({
        action: "split_cluster",
        source_cluster_id: Number(activeClusterId),
        entity_ids: Array.from(selectedEntityIds),
    });
    selectedEntityIds.clear();
    await loadGraph(true);
    const newClusterId = response?.details?.new_cluster_id;
    if (newClusterId !== undefined) {
        selectedClusterIds = new Set([newClusterId]);
        activeClusterId = newClusterId;
        updateSelectionStyles();
        animateNode(newClusterId, "node-split");
        await loadClusterDetails(newClusterId);
    }
};

graphSplitBtn.addEventListener("click", () => {
    openActionModal("split", null);
});

const openContextMenu = (x, y, clusterId) => {
    if (!selectedClusterIds.has(clusterId)) {
        selectedClusterIds = new Set([clusterId]);
        updateSelectionStyles();
    }
    contextTargetId = clusterId;
    contextMenu.style.left = `${x}px`;
    contextMenu.style.top = `${y}px`;
    contextMenu.classList.remove("hidden");
};

const closeContextMenu = () => {
    contextMenu.classList.add("hidden");
    contextTargetId = null;
};

contextMenu.addEventListener("click", (event) => {
    const action = event.target?.dataset?.action;
    if (!action) return;
    openActionModal(action, contextTargetId);
    closeContextMenu();
});

document.addEventListener("click", () => {
    closeContextMenu();
});

document.addEventListener("keydown", (event) => {
    if (!actionModal.classList.contains("hidden")) return;
    if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) {
        return;
    }
    if (event.key.toLowerCase() === "m") {
        if (selectedClusterIds.size >= 2) {
            openActionModal("merge", null);
        }
    }
    if (event.key.toLowerCase() === "s") {
        if (selectedClusterIds.size === 1 && selectedEntityIds.size > 0) {
            openActionModal("split", null);
        }
    }
    if (event.key.toLowerCase() === "n") {
        stepClusterSelection(1);
    }
    if (event.key.toLowerCase() === "p") {
        stepClusterSelection(-1);
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        runUndo();
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        runRedo();
    }
});

const animateNode = (clusterId, className) => {
    const node = nodeElements.get(clusterId);
    if (!node) return;
    node.classList.add(className);
    setTimeout(() => node.classList.remove(className), 700);
};

const positionTooltip = (x, y) => {
    const offset = 12;
    hoverTooltip.style.left = `${x + offset}px`;
    hoverTooltip.style.top = `${y + offset}px`;
};

const startHoverPreview = (clusterId, x, y) => {
    cancelHoverPreview();
    hoverTimer = setTimeout(async () => {
        hoverClusterId = clusterId;
        positionTooltip(x, y);
        tooltipTitle.textContent = `Cluster ${clusterId}`;
        tooltipImage.src = "";
        tooltipImage.style.display = "block";
        tooltipVideo.pause();
        tooltipVideo.src = "";
        tooltipVideo.style.display = "none";
        hoverTooltip.classList.remove("hidden");
        const res = await fetch(`/api/cluster/${clusterId}`);
        if (!res.ok) return;
        const data = await res.json();
        const first = data.items?.[0];
        if (first) {
            if (first.asset_type === "video") {
                tooltipImage.style.display = "none";
                tooltipVideo.style.display = "block";
                tooltipVideo.src = `/media/${first.asset_id}`;
                tooltipVideo.currentTime = 0;
                tooltipVideo.load();
            } else {
                tooltipImage.src = `/media/${first.asset_id}`;
            }
        }
    }, 500);
};

const cancelHoverPreview = () => {
    if (hoverTimer) {
        clearTimeout(hoverTimer);
        hoverTimer = null;
    }
    hoverClusterId = null;
    hoverTooltip.classList.add("hidden");
};

prevNodeBtn.addEventListener("click", () => stepClusterSelection(-1));
nextNodeBtn.addEventListener("click", () => stepClusterSelection(1));
undoBtn.addEventListener("click", runUndo);
redoBtn.addEventListener("click", runRedo);

const openActionModal = (action, targetClusterId) => {
    pendingAction = { action, targetClusterId };
    modalTitle.textContent =
        action === "merge"
            ? "Merge clusters"
            : action === "move"
              ? "Move entities"
              : "Split entities";
    modalDescription.textContent =
        action === "merge"
            ? "Merge selected clusters into a new cluster?"
            : action === "move"
              ? `Move selected entities into Cluster ${targetClusterId}?`
              : "Split selected entities into a new cluster?";
    actionModal.classList.remove("hidden");
};

const closeActionModal = () => {
    actionModal.classList.add("hidden");
    pendingAction = null;
};

modalCancel.addEventListener("click", closeActionModal);

modalConfirm.addEventListener("click", async () => {
    if (!pendingAction) return;
    const { action, targetClusterId } = pendingAction;
    if (action === "merge") {
        await mergeSelectedClusters(targetClusterId);
    } else if (action === "move" && targetClusterId !== null) {
        await moveSelectedEntities(targetClusterId);
    } else if (action === "split") {
        await splitSelectedEntities();
    }
    closeActionModal();
});

clusterSearch.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    const value = Number(clusterSearch.value);
    if (Number.isNaN(value)) return;
    const node = nodes.find((item) => item.cluster_id === value);
    if (node) {
        handleNodeClick(node.cluster_id, {
            shiftKey: false,
            metaKey: false,
            ctrlKey: false,
        });
    }
});

openEditorBtn.addEventListener("click", () => {
    if (activeClusterId === null) {
        window.location.href = "/dashboard";
        return;
    }
    window.location.href = `/dashboard?cluster_id=${activeClusterId}`;
});

let selectionRect = null;
let selectionPath = null;
let lassoPoints = [];
let panState = null;

const startSelection = (event) => {
    if (event.target !== svg) return;
    if (event.altKey) {
        lassoPoints = [toSvgPoint(event.clientX, event.clientY)];
        selectionPath = document.createElementNS(SVG_NS, "path");
        selectionPath.classList.add("selection-lasso");
        overlayGroup.appendChild(selectionPath);
    } else {
        const start = toSvgPoint(event.clientX, event.clientY);
        selectionRect = document.createElementNS(SVG_NS, "rect");
        selectionRect.classList.add("selection-rect");
        selectionRect.dataset.startX = start.x;
        selectionRect.dataset.startY = start.y;
        selectionRect.setAttribute("x", start.x);
        selectionRect.setAttribute("y", start.y);
        selectionRect.setAttribute("width", 0);
        selectionRect.setAttribute("height", 0);
        overlayGroup.appendChild(selectionRect);
    }
};

const startPan = (event) => {
    panState = {
        origin: { x: event.clientX, y: event.clientY },
        viewBox: { ...viewBox },
    };
    svg.style.cursor = "grabbing";
};

const updateSelection = (event) => {
    if (selectionRect) {
        const startX = Number(selectionRect.dataset.startX);
        const startY = Number(selectionRect.dataset.startY);
        const current = toSvgPoint(event.clientX, event.clientY);
        const x = Math.min(startX, current.x);
        const y = Math.min(startY, current.y);
        const w = Math.abs(current.x - startX);
        const h = Math.abs(current.y - startY);
        selectionRect.setAttribute("x", x);
        selectionRect.setAttribute("y", y);
        selectionRect.setAttribute("width", w);
        selectionRect.setAttribute("height", h);
    }
    if (selectionPath) {
        lassoPoints.push(toSvgPoint(event.clientX, event.clientY));
        const path = lassoPoints
            .map(
                (point, idx) =>
                    `${idx === 0 ? "M" : "L"} ${point.x} ${point.y}`,
            )
            .join(" ");
        selectionPath.setAttribute("d", `${path} Z`);
    }
};

const finishSelection = () => {
    if (selectionRect) {
        const x = Number(selectionRect.getAttribute("x"));
        const y = Number(selectionRect.getAttribute("y"));
        const w = Number(selectionRect.getAttribute("width"));
        const h = Number(selectionRect.getAttribute("height"));
        const selected = nodes.filter((node) => {
            const nx = node.x * 1000;
            const ny = node.y * 1000;
            return nx >= x && nx <= x + w && ny >= y && ny <= y + h;
        });
        selectedClusterIds = new Set(selected.map((node) => node.cluster_id));
        updateSelectionStyles();
        if (selected.length === 1) {
            activeClusterId = selected[0].cluster_id;
            loadClusterDetails(activeClusterId);
        } else {
            activeClusterId = null;
            selectedEntityIds.clear();
            assetSelectionMap.clear();
            renderMultiClusterSummary();
        }
        overlayGroup.removeChild(selectionRect);
        selectionRect = null;
    }
    if (selectionPath) {
        const polygon = lassoPoints;
        const contains = (point, poly) => {
            let inside = false;
            for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
                const xi = poly[i].x;
                const yi = poly[i].y;
                const xj = poly[j].x;
                const yj = poly[j].y;
                const intersect =
                    yi > point.y !== yj > point.y &&
                    point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
                if (intersect) inside = !inside;
            }
            return inside;
        };
        const selected = nodes.filter((node) =>
            contains({ x: node.x * 1000, y: node.y * 1000 }, polygon),
        );
        selectedClusterIds = new Set(selected.map((node) => node.cluster_id));
        updateSelectionStyles();
        if (selected.length === 1) {
            activeClusterId = selected[0].cluster_id;
            loadClusterDetails(activeClusterId);
        } else {
            activeClusterId = null;
            selectedEntityIds.clear();
            assetSelectionMap.clear();
            renderMultiClusterSummary();
        }
        overlayGroup.removeChild(selectionPath);
        selectionPath = null;
        lassoPoints = [];
    }
};

svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    if (event.target === svg && event.altKey) {
        startSelection(event);
        return;
    }
    if (event.target === svg && event.shiftKey) {
        startSelection(event);
        return;
    }
    if (event.target === svg) {
        startPan(event);
    }
});

svg.addEventListener("pointermove", (event) => {
    if (panState) {
        const rect = svg.getBoundingClientRect();
        const dx =
            ((event.clientX - panState.origin.x) / rect.width) *
            panState.viewBox.w;
        const dy =
            ((event.clientY - panState.origin.y) / rect.height) *
            panState.viewBox.h;
        viewBox.x = panState.viewBox.x - dx;
        viewBox.y = panState.viewBox.y - dy;
        updateViewBox();
        return;
    }
    if (selectionRect || selectionPath) {
        updateSelection(event);
    }
});

svg.addEventListener("pointerup", () => {
    if (panState) {
        panState = null;
        svg.style.cursor = "grab";
        return;
    }
    if (selectionRect || selectionPath) {
        finishSelection();
    }
});

svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const zoomIntensity = 0.08;
    const delta = Math.sign(event.deltaY);
    const scale = delta > 0 ? 1 + zoomIntensity : 1 - zoomIntensity;
    const mouse = toSvgPoint(event.clientX, event.clientY);
    viewBox.x = mouse.x - (mouse.x - viewBox.x) * scale;
    viewBox.y = mouse.y - (mouse.y - viewBox.y) * scale;
    viewBox.w *= scale;
    viewBox.h *= scale;
    updateViewBox();
});

svg.addEventListener("contextmenu", (event) => {
    event.preventDefault();
});

loadGraph();
