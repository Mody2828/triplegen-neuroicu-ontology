/**
 * Cluster Results — interactive visualization for ontology class clustering.
 * Uses Chart.js for scatter plot + silhouette analysis.
 */
const ClusterResults = (function() {
  // Curated color palette (20 colors, dark-mode optimized)
  const COLORS = [
    '#00d4ff', '#e879f9', '#06efcf', '#fbbf24', '#f87171',
    '#a78bfa', '#34d399', '#fb923c', '#60a5fa', '#f472b6',
    '#22d3ee', '#c084fc', '#4ade80', '#facc15', '#fb7185',
    '#818cf8', '#2dd4bf', '#f97316', '#38bdf8', '#e879f9',
  ];

  let _chart = null;  // Chart.js instance
  let _silChart = null;

  function open(runId) {
    const container = document.getElementById('cluster-results-container');
    const btn = document.getElementById('btn-cluster-results');
    if (!container) return;

    // Toggle visibility
    if (container.style.display === 'block') {
      container.style.display = 'none';
      return;
    }

    container.style.display = 'block';
    container.innerHTML = `
      <div class="text-center py-5">
        <div class="spinner-border text-info" role="status" style="width:3rem;height:3rem;"></div>
        <p class="mt-3 text-muted">Clustering classes with semantic embeddings...</p>
        <p class="text-muted small">This may take a few seconds on first run (loading model)</p>
      </div>
    `;

    fetch(`/api/run/${runId}/cluster`, { method: 'POST' })
      .then(r => r.json())
      .then(data => {
        if (data.error) {
          container.innerHTML = `<div class="alert alert-warning">${data.error}</div>`;
          return;
        }
        render(container, data);
      })
      .catch(err => {
        container.innerHTML = `<div class="alert alert-danger">Clustering failed: ${err.message}</div>`;
      });
  }

  function render(container, data) {
    container.innerHTML = `
      <!-- Summary stats -->
      <div class="row mb-4">
        <div class="col-md-3">
          <div class="cluster-stat-card">
            <div class="stat-value">${data.total_classes}</div>
            <div class="stat-label">Total Classes</div>
          </div>
        </div>
        <div class="col-md-3">
          <div class="cluster-stat-card">
            <div class="stat-value">${data.optimal_k}</div>
            <div class="stat-label">Clusters Found</div>
          </div>
        </div>
        <div class="col-md-3">
          <div class="cluster-stat-card">
            <div class="stat-value">${data.silhouette_score.toFixed(4)}</div>
            <div class="stat-label">Silhouette Score</div>
          </div>
        </div>
        <div class="col-md-3">
          <div class="cluster-stat-card">
            <div class="stat-value">${(data.pca_variance[0]*100).toFixed(1)}%</div>
            <div class="stat-label">PCA Variance (PC1)</div>
          </div>
        </div>
      </div>

      <!-- Charts row -->
      <div class="row mb-4">
        <div class="col-lg-8">
          <div class="cluster-chart-wrap">
            <canvas id="cluster-scatter-chart"></canvas>
          </div>
        </div>
        <div class="col-lg-4">
          <div class="cluster-chart-wrap">
            <canvas id="cluster-sil-chart"></canvas>
          </div>
        </div>
      </div>

      <!-- Cluster cards -->
      <div class="row" id="cluster-cards-row"></div>
    `;

    renderScatter(data);
    renderSilhouette(data);
    renderCards(data);
  }

  function renderScatter(data) {
    const ctx = document.getElementById('cluster-scatter-chart');
    if (!ctx) return;

    // Group points by cluster
    const datasets = [];
    const grouped = {};
    data.pca_points.forEach(p => {
      if (!grouped[p.cluster]) grouped[p.cluster] = { name: p.cluster_name, points: [] };
      grouped[p.cluster].points.push(p);
    });

    Object.keys(grouped).sort((a,b) => +a - +b).forEach(cid => {
      const g = grouped[cid];
      const color = COLORS[cid % COLORS.length];
      datasets.push({
        label: `C${cid}: ${g.name.substring(0, 30)}`,
        data: g.points.map(p => ({ x: p.x, y: p.y, label: p.label })),
        backgroundColor: color + 'cc',
        borderColor: color,
        borderWidth: 1,
        pointRadius: 5,
        pointHoverRadius: 8,
      });
    });

    if (_chart) _chart.destroy();
    _chart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Ontology Class Clusters — Semantic Embeddings (PCA)',
            color: '#e2eaf4',
            font: { size: 15, weight: 'bold' },
          },
          legend: {
            display: true,
            position: 'right',
            labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 12, padding: 6 },
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const pt = context.raw;
                return pt.label || '';
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: `PC1 (${(data.pca_variance[0]*100).toFixed(1)}%)`, color: '#94a3b8' },
            ticks: { color: '#94a3b8' },
            grid: { color: '#1e293b' },
          },
          y: {
            title: { display: true, text: `PC2 (${(data.pca_variance[1]*100).toFixed(1)}%)`, color: '#94a3b8' },
            ticks: { color: '#94a3b8' },
            grid: { color: '#1e293b' },
          }
        },
      }
    });
  }

  function renderSilhouette(data) {
    const ctx = document.getElementById('cluster-sil-chart');
    if (!ctx || !data.silhouette_curve) return;

    const curve = data.silhouette_curve;

    if (_silChart) _silChart.destroy();
    _silChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: curve.map(d => d.k),
        datasets: [{
          label: 'Silhouette Score',
          data: curve.map(d => d.score),
          borderColor: '#00d4ff',
          backgroundColor: 'rgba(0,212,255,0.15)',
          fill: true,
          borderWidth: 2.5,
          pointBackgroundColor: '#06efcf',
          pointBorderColor: '#fff',
          pointRadius: 5,
          tension: 0.3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Optimal k (Silhouette)',
            color: '#e2eaf4',
            font: { size: 14, weight: 'bold' },
          },
          legend: { display: false },
          annotation: {
            annotations: {
              bestLine: {
                type: 'line',
                xMin: data.optimal_k,
                xMax: data.optimal_k,
                borderColor: '#e879f9',
                borderWidth: 2,
                borderDash: [6, 3],
                label: {
                  display: true,
                  content: `Best k=${data.optimal_k}`,
                  color: '#e879f9',
                  backgroundColor: 'rgba(26,26,46,0.9)',
                }
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'k', color: '#94a3b8' },
            ticks: { color: '#94a3b8' },
            grid: { color: '#1e293b' },
          },
          y: {
            title: { display: true, text: 'Score', color: '#94a3b8' },
            ticks: { color: '#94a3b8' },
            grid: { color: '#1e293b' },
          }
        }
      }
    });
  }

  function renderCards(data) {
    const row = document.getElementById('cluster-cards-row');
    if (!row) return;

    let html = '';
    data.clusters.forEach(cl => {
      const color = COLORS[cl.id % COLORS.length];
      const memberPills = cl.members.map(m =>
        `<span class="cluster-member-pill">${m}</span>`
      ).join('');

      html += `
        <div class="col-lg-4 col-md-6 mb-3">
          <div class="cluster-card" style="border-left-color: ${color};">
            <div class="cluster-card-header">
              <span class="cluster-card-id" style="color: ${color};">C${cl.id}</span>
              <span class="cluster-card-name">${cl.name}</span>
              <span class="cluster-card-count">${cl.size}</span>
            </div>
            <div class="cluster-card-members">${memberPills}</div>
          </div>
        </div>
      `;
    });
    row.innerHTML = html;
  }

  return { open };
})();
