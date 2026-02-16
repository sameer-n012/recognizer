async function refreshClusters() {
    const res = await fetch("/api/refresh");
    if (res.ok) {
        document.getElementById("cluster-list").innerHTML = "";
        document.getElementById("grid").innerHTML = "";
        document.getElementById("cluster-title").textContent =
            "Select a cluster";
        await loadClusters();
    }
}

async function loadClusters() {
    const res = await fetch("/api/clusters");
    const data = await res.json();
    const list = document.getElementById("cluster-list");

    data.clusters.forEach((c) => {
        const li = document.createElement("li");
        li.textContent = `Cluster ${c.cluster_id} (${c.size})`;
        li.onclick = () => loadCluster(c.cluster_id);
        list.appendChild(li);
    });
}

async function loadCluster(clusterId) {
    const res = await fetch(`/api/cluster/${clusterId}`);
    const data = await res.json();

    document.getElementById("cluster-title").textContent =
        `Cluster ${clusterId}`;

    const grid = document.getElementById("grid");
    grid.innerHTML = "";

    data.items.forEach((item) => {
        const card = document.createElement("div");
        card.className = "card";

        const url = `/media/${item.asset_id}`;
        console.log(url);
        // if (url.match(/\.(mp4|mov|webm|m4v|mkv|avi)$/)) {
        if (item.asset_type === "video") {
            const video = document.createElement("video");
            video.src = url;
            video.controls = true;
            card.appendChild(video);
        } else {
            const img = document.createElement("img");
            img.src = url;
            card.appendChild(img);
        }

        card.onclick = () => {
            window.open(url, "_blank");
        };

        grid.appendChild(card);
    });
}

loadClusters();
