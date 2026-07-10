const GROUP_PAGE_LIMIT = 24;
const MEMBER_PAGE_LIMIT = 12;

const state = {
    selected: new Set(),
    showSingletons: false,
    duplicateBucketCount: 0,
    singletonBucketCount: 0,
    duplicateAssetCount: 0,
    groupOffset: 0,
    hasMoreGroups: true,
    loadingGroups: false,
    // groupId -> { offset, hasMore, loading }, tracks each bucket's own member paging
    // independently of the outer bucket-list paging.
    memberPaging: new Map(),
};

const bucketGrid = document.getElementById("bucket-grid");
const groupCountPill = document.getElementById("group-count-pill");
const assetCountPill = document.getElementById("asset-count-pill");
const singletonCountPill = document.getElementById("singleton-count-pill");
const singletonToggle = document.getElementById("singleton-toggle");
const mergeButton = document.getElementById("merge-button");
const undoButton = document.getElementById("undo-button");
const redoButton = document.getElementById("redo-button");
const refreshButton = document.getElementById("refresh-button");

let groupSentinel = null;

async function fetchJSON(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Request failed: ${res.status}`);
    }
    return res.json();
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function mediaTag(member) {
    const src = `/media/${encodeURIComponent(member.asset_id)}`;
    const basename = escapeHtml(member.basename);
    const inner =
        member.type === "video"
            ? `<video src="${src}#t=0.1" preload="metadata" muted playsinline></video>`
            : `<img src="${src}" loading="lazy" alt="${basename}" />`;
    return `<a href="${src}" target="_blank" rel="noopener" title="Open ${basename} in a new tab">${inner}</a>`;
}

function bucketTitle(group) {
    // Real duplicate buckets get a stable, meaningful id from the pipeline; ephemeral
    // "unmatched" singleton buckets get an arbitrary negative id purely so they're
    // selectable (see add_unmatched_singletons in server.py) — showing that raw
    // negative number ("Bucket -3") reads as a bug rather than an id, so show the
    // filename instead for any singleton bucket.
    if (group.is_singleton && group.members[0]) {
        const basename = escapeHtml(group.members[0].basename);
        return basename.length > 10 ? `${basename.substring(0, 10)}...` : basename;
    }
    return `Bucket ${group.group_id}`;
}

// ---- Bucket-list (outer) infinite scroll ------------------------------------------

const groupListObserver = new IntersectionObserver(
    (entries) => {
        if (entries[0].isIntersecting) loadMoreGroups();
    },
    { rootMargin: "600px" },
);

function ensureGroupSentinel() {
    if (groupSentinel) {
        groupListObserver.unobserve(groupSentinel);
        groupSentinel.remove();
    }
    groupSentinel = document.createElement("div");
    groupSentinel.className = "scroll-sentinel";
    bucketGrid.appendChild(groupSentinel);
    groupListObserver.observe(groupSentinel);
}

async function resetAndLoadGroups() {
    groupListObserver.disconnect();
    memberObserver.disconnect();
    state.groupOffset = 0;
    state.hasMoreGroups = true;
    state.loadingGroups = false;
    state.memberPaging.clear();
    bucketGrid.innerHTML = "";
    groupSentinel = null;
    await loadMoreGroups();
}

async function loadMoreGroups() {
    if (state.loadingGroups || !state.hasMoreGroups) return;
    state.loadingGroups = true;
    try {
        const params = new URLSearchParams({
            offset: state.groupOffset,
            limit: GROUP_PAGE_LIMIT,
            member_limit: MEMBER_PAGE_LIMIT,
        });
        if (state.showSingletons) params.set("include_singletons", "1");

        const data = await fetchJSON(`/api/groups?${params}`);
        state.duplicateBucketCount = data.duplicate_bucket_count;
        state.singletonBucketCount = data.singleton_bucket_count;
        state.duplicateAssetCount = data.duplicate_asset_count;
        state.hasMoreGroups = data.has_more;
        state.groupOffset = data.offset + data.groups.length;

        if (groupSentinel) groupSentinel.remove();

        if (data.offset === 0 && data.groups.length === 0) {
            bucketGrid.innerHTML = state.showSingletons
                ? '<p class="placeholder">Nothing to show.</p>'
                : '<p class="placeholder">No duplicate buckets found. Run the pipeline through cluster_duplicates first.</p>';
        } else {
            for (const group of data.groups) {
                state.memberPaging.set(group.group_id, {
                    offset: group.members.length,
                    hasMore: group.has_more_members,
                    loading: false,
                });
                bucketGrid.appendChild(renderBucket(group));
            }
        }

        if (state.hasMoreGroups) {
            ensureGroupSentinel();
        }

        updatePills();
    } catch (err) {
        bucketGrid.innerHTML = `<p class="placeholder">Failed to load buckets: ${err.message}</p>`;
    } finally {
        state.loadingGroups = false;
    }
}

function updatePills() {
    groupCountPill.textContent = `Buckets • ${state.duplicateBucketCount}`;
    assetCountPill.textContent = `Assets • ${state.duplicateAssetCount}`;
    singletonCountPill.textContent = `Singletons • ${state.singletonBucketCount}`;
    mergeButton.disabled = state.selected.size < 2;
}

// ---- Per-bucket (inner) member infinite scroll -------------------------------------

const memberObserver = new IntersectionObserver(
    (entries) => {
        for (const entry of entries) {
            if (entry.isIntersecting) {
                loadMoreMembers(Number(entry.target.dataset.groupId));
            }
        }
    },
    { rootMargin: "300px" },
);

async function loadMoreMembers(groupId) {
    const paging = state.memberPaging.get(groupId);
    if (!paging || paging.loading || !paging.hasMore) return;
    paging.loading = true;

    try {
        const params = new URLSearchParams({
            offset: paging.offset,
            limit: MEMBER_PAGE_LIMIT,
        });
        const data = await fetchJSON(`/api/group/${groupId}/members?${params}`);

        const card = bucketGrid.querySelector(`[data-group-id="${groupId}"]`);
        if (!card) return;
        const items = card.querySelector(".bucket-items");
        const sentinel = items.querySelector(".scroll-sentinel");

        for (const member of data.members) {
            // Only buckets with size > 1 can ever have has_more_members true (a
            // singleton's total is always 1, so pagination past it never triggers) —
            // safe to always show remove here.
            items.insertBefore(renderThumb(member, true), sentinel);
        }

        paging.offset = data.member_offset + data.members.length;
        paging.hasMore = data.has_more_members;

        if (sentinel) {
            if (!paging.hasMore) {
                memberObserver.unobserve(sentinel);
                sentinel.remove();
            }
            // If paging.hasMore is still true, the sentinel stays put and remains
            // observed (still in the DOM, position unchanged) — no action needed.
        }

        updatePills();
    } finally {
        paging.loading = false;
    }
}

function renderThumb(member, showRemove) {
    const thumb = document.createElement("div");
    thumb.className = "item-thumb";
    thumb.innerHTML = mediaTag(member);
    if (member.is_keeper) {
        const badge = document.createElement("span");
        badge.className = "keeper-badge";
        badge.textContent = "keeper";
        thumb.appendChild(badge);
    }
    if (showRemove) {
        const removeBtn = document.createElement("button");
        removeBtn.className = "item-remove";
        removeBtn.textContent = "✕";
        removeBtn.title = "Remove from bucket";
        removeBtn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            removeAsset(member.asset_id);
        });
        thumb.appendChild(removeBtn);
    }
    return thumb;
}

function renderBucket(group) {
    const card = document.createElement("article");
    card.className = "bucket-card";
    if (group.is_singleton) card.classList.add("singleton");
    card.dataset.groupId = group.group_id;
    if (state.selected.has(group.group_id)) {
        card.classList.add("selected");
    }

    const head = document.createElement("div");
    head.className = "bucket-head";
    head.innerHTML = `
        <label>
            <input type="checkbox" class="bucket-select" ${state.selected.has(group.group_id) ? "checked" : ""} />
            <span class="bucket-title">${bucketTitle(group)}</span>
        </label>
        <span class="bucket-size">${group.size} item${group.size === 1 ? "" : "s"}</span>
    `;
    head.querySelector(".bucket-select").addEventListener("change", (e) => {
        toggleSelection(group.group_id, e.target.checked);
    });
    card.appendChild(head);

    const items = document.createElement("div");
    items.className = "bucket-items";
    for (const member of group.members) {
        items.appendChild(renderThumb(member, group.size > 1));
    }
    if (group.has_more_members) {
        const sentinel = document.createElement("div");
        sentinel.className = "scroll-sentinel member-sentinel";
        sentinel.dataset.groupId = group.group_id;
        items.appendChild(sentinel);
        memberObserver.observe(sentinel);
    }
    card.appendChild(items);

    return card;
}

function toggleSelection(groupId, checked) {
    if (checked) {
        state.selected.add(groupId);
    } else {
        state.selected.delete(groupId);
    }
    const card = bucketGrid.querySelector(`[data-group-id="${groupId}"]`);
    if (card) card.classList.toggle("selected", checked);
    mergeButton.disabled = state.selected.size < 2;
}

async function mergeSelected() {
    if (state.selected.size < 2) return;
    try {
        await fetchJSON("/api/group/merge", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ group_ids: [...state.selected] }),
        });
        state.selected.clear();
        await resetAndLoadGroups();
    } catch (err) {
        alert(err.message);
    }
}

async function removeAsset(assetId) {
    try {
        await fetchJSON("/api/group/remove", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ asset_id: assetId }),
        });
        await resetAndLoadGroups();
    } catch (err) {
        alert(err.message);
    }
}

async function undo() {
    try {
        await fetchJSON("/api/group/undo", { method: "POST" });
        await resetAndLoadGroups();
    } catch (err) {
        alert(err.message);
    }
}

async function redo() {
    try {
        await fetchJSON("/api/group/redo", { method: "POST" });
        await resetAndLoadGroups();
    } catch (err) {
        alert(err.message);
    }
}

async function refresh() {
    await fetchJSON("/api/refresh");
    await resetAndLoadGroups();
}

mergeButton.addEventListener("click", mergeSelected);
undoButton.addEventListener("click", undo);
redoButton.addEventListener("click", redo);
refreshButton.addEventListener("click", refresh);
singletonToggle.addEventListener("change", (e) => {
    state.showSingletons = e.target.checked;
    resetAndLoadGroups().catch((err) => alert(err.message));
});

resetAndLoadGroups().catch((err) => {
    bucketGrid.innerHTML = `<p class="placeholder">Failed to load buckets: ${err.message}</p>`;
});
