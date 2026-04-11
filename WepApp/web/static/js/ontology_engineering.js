/**
 * Ontology Engineering — 3-stage pipeline: Merge → Cluster → Reconstruct.
 */
(function () {
  'use strict';

  // ── Color palette (matches cluster_results.js) ──
  const COLORS = [
    '#00d4ff', '#e879f9', '#06efcf', '#fbbf24', '#f87171',
    '#a78bfa', '#34d399', '#fb923c', '#60a5fa', '#f472b6',
    '#22d3ee', '#c084fc', '#4ade80', '#facc15', '#fb7185',
    '#818cf8', '#2dd4bf', '#f97316', '#38bdf8', '#e879f9',
  ];

  // ── State ──
  let sessionId = null;
  let scatterChart = null;
  let silChart = null;
  let cyInstance = null;
  let sourceRunIds = [];         // run_ids selected in Stage 1 (for Compare Metrics)
  let reconstructedRunId = null; // run_id produced by Stage 3 reconstruction

  // ── Helpers ──
  function $(id) { return document.getElementById(id); }

  function unlock(stageEl, stepEl) {
    stageEl.classList.remove('locked');
    stepEl.classList.add('active');
  }

  function markDone(stepEl) {
    stepEl.classList.remove('active');
    stepEl.classList.add('done');
  }

  function statCard(value, label) {
    return `<div class="oe-stat-card">
      <div class="oe-stat-value">${value}</div>
      <div class="oe-stat-label">${label}</div>
    </div>`;
  }

  // ── Select all / deselect all ──
  const checkboxes = () => Array.from(document.querySelectorAll('.oe-run-check'));
  const updateCount = () => {
    const n = checkboxes().filter(c => c.checked).length;
    const el = $('selected-count');
    if (el) el.textContent = n + ' selected';
    const btnText = $('btn-merge-text');
    if (btnText) btnText.textContent = n > 1 ? 'Merge Selected' : 'Load Selected';
    const anText = $('btn-oe-analyze-text');
    if (anText) anText.textContent = n > 1 ? 'Analyze Selected' : 'Analyze';
  };

  if ($('btn-select-all')) {
    $('btn-select-all').addEventListener('click', () => { checkboxes().forEach(c => c.checked = true); updateCount(); });
  }
  if ($('btn-deselect-all')) {
    $('btn-deselect-all').addEventListener('click', () => { checkboxes().forEach(c => c.checked = false); updateCount(); });
  }
  document.querySelectorAll('.oe-run-check').forEach(c => c.addEventListener('change', updateCount));
  updateCount();

  // ══════════════════════════════════════════════════════════════════
  //  Last-result + Analyze (selected) — mirrors the run comparison page
  // ══════════════════════════════════════════════════════════════════

  function escHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function fmt2(v) { return (v == null) ? '—' : (Math.round(v * 100) / 100); }

  // Per-row "Last result" button — opens the single-run summary modal.
  document.querySelectorAll('.oe-run-view-result').forEach(btn => {
    btn.addEventListener('click', function (ev) {
      ev.preventDefault();
      ev.stopPropagation();
      const runId = this.getAttribute('data-run-id');
      if (!runId) return;
      const body = $('oe-result-modal-body');
      const link = $('oe-result-modal-link');
      const modalEl = $('oe-result-modal');
      if (!body || !modalEl) return;
      body.innerHTML = '<p class="text-muted mb-0">Loading…</p>';
      if (link) link.style.display = 'none';
      bootstrap.Modal.getOrCreateInstance(modalEl).show();
      fetch('/api/run/' + encodeURIComponent(runId) + '/summary')
        .then(r => r.json())
        .then(data => {
          if (data && data.found) {
            body.innerHTML = '<pre class="metrics-pre mb-0" style="white-space: pre-wrap;">'
              + escHtml(data.summary || '') + '</pre>';
            if (link) {
              link.href = '/results/' + encodeURIComponent(data.run_id || runId);
              link.style.display = 'inline-block';
            }
          } else {
            body.innerHTML = '<p class="text-muted mb-0">'
              + escHtml((data && data.message) || 'No data saved for this run.') + '</p>';
          }
        })
        .catch(() => {
          body.innerHTML = '<p class="text-danger mb-0">Failed to load result.</p>';
        });
    });
  });

  // Analyze / Analyze Selected — compare metrics across selected runs.
  let _oeAnalyzeChart = null;
  if ($('btn-oe-analyze')) {
    $('btn-oe-analyze').addEventListener('click', function () {
      const selected = checkboxes().filter(c => c.checked).map(c => c.value);
      if (!selected.length) { alert('Select at least one run.'); return; }
      const modalEl = $('oe-analyze-modal');
      const body = $('oe-analyze-modal-body');
      const titleEl = $('oe-analyze-modal-title');
      if (!modalEl || !body) return;
      if (titleEl) titleEl.textContent = selected.length > 1 ? 'Analyze selected' : 'Analyze';
      body.innerHTML = '<p class="text-muted mb-0">Loading…</p>';
      bootstrap.Modal.getOrCreateInstance(modalEl).show();

      fetch('/api/runs-metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_ids: selected }),
      })
        .then(r => r.json())
        .then(data => renderAnalyzeModal(data && data.rows ? data.rows : [], body))
        .catch(() => { body.innerHTML = '<p class="text-danger mb-0">Failed to load analysis.</p>'; });
    });
  }

  // opts = { labelOverrides: { [run_id]: 'Display label' }, emptyMessage: '...' }
  function renderAnalyzeModal(rows, body, opts) {
    opts = opts || {};
    if (opts.labelOverrides) {
      rows = rows.map(r => {
        const lbl = opts.labelOverrides[r.run_id];
        return lbl ? Object.assign({}, r, { label: lbl }) : r;
      });
    }
    const withData = rows.filter(r => r.run_id != null && r.f1 != null);
    if (!withData.length) {
      body.innerHTML = '<p class="text-muted mb-0">'
        + escHtml(opts.emptyMessage || 'No evaluation data saved for the selected run(s). Run them first (or ensure metrics.json exists), then try again.')
        + '</p>';
      return;
    }

    // Sort by Overall F1 desc so the best run is at the top (class+hier+rel mean).
    // Falls back to class F1 if overall_f1 is missing (legacy metrics files).
    const _sortKey = r => (r.overall_f1 != null ? r.overall_f1 : (r.f1 != null ? r.f1 : -1));
    const sorted = rows.slice().sort((a, b) => _sortKey(b) - _sortKey(a));
    const chartRows = sorted.filter(r => r.run_id != null && r.f1 != null);
    const chartLabels = chartRows.map(r => r.label || r.run_id);

    // Summary cards (single-run shows a compact stat strip; multi shows best/avg/etc.)
    let cardsHtml = '';
    if (chartRows.length === 1) {
      const r0 = chartRows[0];
      // Two rows of cards so structural F1s are visible alongside class metrics.
      cardsHtml = '<div class="row g-2 mb-2">'
        + '<div class="col"><div class="card summary-card border-success"><div class="card-body py-2"><div class="small text-muted">Overall F1</div><div class="fw-bold">' + fmt2(r0.overall_f1) + '</div><div class="small text-muted">mean of class+hier+rel</div></div></div></div>'
        + '<div class="col"><div class="card summary-card"><div class="card-body py-2"><div class="small text-muted">Class F1</div><div class="fw-bold">' + fmt2(r0.f1) + '</div></div></div></div>'
        + '<div class="col"><div class="card summary-card"><div class="card-body py-2"><div class="small text-muted">Hierarchy F1</div><div class="fw-bold">' + fmt2(r0.hier_f1) + '</div></div></div></div>'
        + '<div class="col"><div class="card summary-card"><div class="card-body py-2"><div class="small text-muted">Relation F1</div><div class="fw-bold">' + fmt2(r0.rel_f1) + '</div></div></div></div>'
        + '</div>'
        + '<div class="row g-2 mb-3">'
        + '<div class="col"><div class="card summary-card"><div class="card-body py-2"><div class="small text-muted">Class Precision</div><div class="fw-bold">' + fmt2(r0.precision) + '</div></div></div></div>'
        + '<div class="col"><div class="card summary-card"><div class="card-body py-2"><div class="small text-muted">Class Recall</div><div class="fw-bold">' + fmt2(r0.recall) + '</div></div></div></div>'
        + '<div class="col"><div class="card summary-card"><div class="card-body py-2"><div class="small text-muted">Coverage</div><div class="fw-bold">' + fmt2(r0.coverage) + '</div></div></div></div>'
        + '</div>';
    } else {
      // For multi-run, show best Overall F1 + best of each sub-metric.
      let bestOverall = chartRows[0], bestClass = chartRows[0], bestHier = chartRows[0], bestRel = chartRows[0];
      let sumOverall = 0, cntOverall = 0;
      chartRows.forEach(x => {
        if ((x.overall_f1 || 0) > (bestOverall.overall_f1 || 0)) bestOverall = x;
        if ((x.f1 || 0) > (bestClass.f1 || 0)) bestClass = x;
        if ((x.hier_f1 || 0) > (bestHier.hier_f1 || 0)) bestHier = x;
        if ((x.rel_f1 || 0) > (bestRel.rel_f1 || 0)) bestRel = x;
        if (x.overall_f1 != null) { sumOverall += x.overall_f1; cntOverall++; }
      });
      const avgOverall = cntOverall ? (Math.round((sumOverall / cntOverall) * 100) / 100) : 0;
      cardsHtml = '<div class="row g-2 mb-3">'
        + '<div class="col"><div class="card summary-card border-success"><div class="card-body py-2"><div class="small text-muted">Best Overall F1</div><div class="fw-bold">' + fmt2(bestOverall.overall_f1) + '</div><div class="small text-truncate" title="' + escHtml(bestOverall.label) + '">' + escHtml(bestOverall.label) + '</div></div></div></div>'
        + '<div class="col"><div class="card summary-card"><div class="card-body py-2"><div class="small text-muted">Best Class F1</div><div class="fw-bold">' + fmt2(bestClass.f1) + '</div><div class="small text-truncate">' + escHtml(bestClass.label) + '</div></div></div></div>'
        + '<div class="col"><div class="card summary-card"><div class="card-body py-2"><div class="small text-muted">Best Hierarchy F1</div><div class="fw-bold">' + fmt2(bestHier.hier_f1) + '</div><div class="small text-truncate">' + escHtml(bestHier.label) + '</div></div></div></div>'
        + '<div class="col"><div class="card summary-card"><div class="card-body py-2"><div class="small text-muted">Best Relation F1</div><div class="fw-bold">' + fmt2(bestRel.rel_f1) + '</div><div class="small text-truncate">' + escHtml(bestRel.label) + '</div></div></div></div>'
        + '<div class="col"><div class="card summary-card"><div class="card-body py-2"><div class="small text-muted">Avg Overall F1</div><div class="fw-bold">' + avgOverall + '</div><div class="small">' + cntOverall + ' runs</div></div></div></div>'
        + '</div>';
    }

    const wrapClass = chartRows.length > 15 ? ' chart-many-runs' : '';
    const chartHtml = '<div id="oe-analyze-chart-wrap" class="' + wrapClass.trim() + '"><canvas id="oe-analyze-chart"></canvas></div>';

    // Sortable table — now showing structural F1s as well as class metrics.
    // "Overall F1" is the mean of (class, hierarchy, relation) F1 and is the
    // primary ranking signal for reconstruction-vs-source comparisons.
    let tableHtml = '<div class="table-responsive"><table class="table table-sm table-hover">'
      + '<thead><tr>'
      + '<th>Run</th>'
      + '<th title="Mean of Class F1, Hierarchy F1, Relation F1">Overall F1</th>'
      + '<th>Class F1</th>'
      + '<th>Hier F1</th>'
      + '<th>Rel F1</th>'
      + '<th>Precision</th>'
      + '<th>Recall</th>'
      + '<th></th>'
      + '</tr></thead><tbody>';
    sorted.forEach((row, i) => {
      const label = row.label || row.run_id || ('Run ' + (i + 1));
      if (row.run_id == null || row.f1 == null) {
        tableHtml += '<tr><td>' + escHtml(label) + '</td>'
          + '<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>'
          + '<td><span class="text-muted small">No data</span></td></tr>';
        return;
      }
      const badge = (i === 0 && chartRows.length > 1) ? ' <span class="badge bg-success">best</span>' : '';
      const viewHref = '/results/' + encodeURIComponent(row.run_id);
      tableHtml += '<tr>'
        + '<td>' + escHtml(label) + badge + '</td>'
        + '<td><strong>' + fmt2(row.overall_f1) + '</strong></td>'
        + '<td>' + fmt2(row.f1) + '</td>'
        + '<td>' + fmt2(row.hier_f1) + '</td>'
        + '<td>' + fmt2(row.rel_f1) + '</td>'
        + '<td>' + fmt2(row.precision) + '</td>'
        + '<td>' + fmt2(row.recall) + '</td>'
        + '<td><a href="' + viewHref + '" class="btn btn-outline-tech btn-sm" target="_blank" rel="noopener">'
        + '<i class="bi bi-box-arrow-up-right me-1"></i>View</a></td>'
        + '</tr>';
    });
    tableHtml += '</tbody></table></div>';

    body.innerHTML = cardsHtml + chartHtml + tableHtml;

    // Chart.js bar chart — Overall / Class / Hierarchy / Relation F1 side-by-side.
    // Replaces the previous F1/P/R triple so structural improvements (hierarchy
    // and relation F1) are visible alongside class-level F1. Class precision and
    // recall are still available as table columns and as cards for single-run view.
    if (typeof Chart !== 'undefined' && chartRows.length) {
      if (_oeAnalyzeChart) { _oeAnalyzeChart.destroy(); _oeAnalyzeChart = null; }
      const ctx = document.getElementById('oe-analyze-chart');
      if (ctx) {
        const _round = v => (v != null ? Math.round(v * 100) / 100 : 0);
        const overallVals = chartRows.map(r => _round(r.overall_f1));
        const classVals   = chartRows.map(r => _round(r.f1));
        const hierVals    = chartRows.map(r => _round(r.hier_f1));
        const relVals     = chartRows.map(r => _round(r.rel_f1));
        _oeAnalyzeChart = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: chartLabels,
            datasets: [
              { label: 'Overall F1',   data: overallVals, backgroundColor: 'rgba(46, 214, 161, 0.9)', borderColor: '#2ED6A1', borderWidth: 1, borderRadius: 4 },
              { label: 'Class F1',     data: classVals,   backgroundColor: 'rgba(57, 198, 230, 0.9)', borderColor: '#39C6E6', borderWidth: 1, borderRadius: 4 },
              { label: 'Hierarchy F1', data: hierVals,    backgroundColor: 'rgba(245, 166, 91, 0.9)', borderColor: '#F5A65B', borderWidth: 1, borderRadius: 4 },
              { label: 'Relation F1',  data: relVals,     backgroundColor: 'rgba(196, 133, 255, 0.9)', borderColor: '#C485FF', borderWidth: 1, borderRadius: 4 },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            color: '#EAF6FF',
            scales: {
              y: { min: 0, max: 1, ticks: { color: '#7FA2BA', font: { size: 11 }, callback: v => (v * 100) + '%' }, grid: { color: 'rgba(120,170,210,0.12)' } },
              x: { grid: { display: false }, ticks: { color: '#7FA2BA', font: { size: 10 }, maxRotation: 45, minRotation: 0 } },
            },
            plugins: {
              legend: { position: 'top', labels: { color: '#A8C4D8', font: { size: 12 } } },
              tooltip: {
                backgroundColor: '#0B1E33', borderColor: '#2A5D86', borderWidth: 1,
                titleColor: '#EAF6FF', bodyColor: '#A8C4D8',
                callbacks: { label: ctx => ctx.dataset.label + ': ' + (ctx.raw * 100).toFixed(1) + '%' },
              },
            },
          },
        });
      }
    }
  }

  // Destroy the chart when the analyze modal is closed to avoid Chart.js resize loops.
  const _oeAnalyzeModalEl = document.getElementById('oe-analyze-modal');
  if (_oeAnalyzeModalEl) {
    _oeAnalyzeModalEl.addEventListener('hidden.bs.modal', () => {
      if (_oeAnalyzeChart) { _oeAnalyzeChart.destroy(); _oeAnalyzeChart = null; }
    });
  }

  // ══════════════════════════════════════════════════════════════════
  //  Stage 1: Merge
  // ══════════════════════════════════════════════════════════════════
  if ($('btn-merge')) {
    $('btn-merge').addEventListener('click', function () {
      const selected = checkboxes().filter(c => c.checked).map(c => c.value);
      if (selected.length < 1) {
        alert('Select at least one run.');
        return;
      }

      $('merge-spinner').classList.remove('d-none');
      $('btn-merge').disabled = true;

      fetch('/api/ontology-engineering/merge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_ids: selected }),
      })
        .then(r => r.json())
        .then(data => {
          $('merge-spinner').classList.add('d-none');
          $('btn-merge').disabled = false;

          if (data.error) {
            alert('Merge failed: ' + data.error);
            return;
          }

          sessionId = data.session_id;
          sourceRunIds = selected.slice();   // remember for Compare Metrics in Stage 3
          reconstructedRunId = null;         // reset any prior reconstruction state
          const s = data.stats;
          const srcLabel = selected.length > 1 ? 'Source Runs' : 'Source Run';

          $('merge-stats').innerHTML =
            statCard(s.total_classes, 'Classes') +
            statCard(s.total_relations, 'Relations') +
            statCard(s.total_hierarchy, 'Hierarchy Edges') +
            statCard(selected.length, srcLabel);
          $('merge-results').classList.remove('d-none');

          // Advance stepper
          markDone($('step-1'));
          unlock($('stage-cluster'), $('step-2'));
        })
        .catch(err => {
          $('merge-spinner').classList.add('d-none');
          $('btn-merge').disabled = false;
          alert('Merge error: ' + err.message);
        });
    });
  }

  // ══════════════════════════════════════════════════════════════════
  //  Stage 2: Cluster
  // ══════════════════════════════════════════════════════════════════
  if ($('btn-cluster')) {
    $('btn-cluster').addEventListener('click', function () {
      if (!sessionId) { alert('Merge first.'); return; }

      $('cluster-spinner').classList.remove('d-none');
      $('btn-cluster').disabled = true;

      fetch('/api/ontology-engineering/cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      })
        .then(r => r.json())
        .then(data => {
          $('cluster-spinner').classList.add('d-none');
          $('btn-cluster').disabled = false;

          if (data.error) {
            alert('Clustering failed: ' + data.error);
            return;
          }

          renderClusterResults(data);
          $('cluster-results').classList.remove('d-none');

          // Advance stepper
          markDone($('step-2'));
          unlock($('stage-reconstruct'), $('step-3'));
        })
        .catch(err => {
          $('cluster-spinner').classList.add('d-none');
          $('btn-cluster').disabled = false;
          alert('Cluster error: ' + err.message);
        });
    });
  }

  function renderClusterResults(data) {
    // Stats
    $('cluster-stats').innerHTML =
      statCard(data.total_classes, 'Total Classes') +
      statCard(data.optimal_k, 'Clusters') +
      statCard(data.silhouette_score.toFixed(4), 'Silhouette') +
      statCard((data.pca_variance[0] * 100).toFixed(1) + '%', 'PC1 Variance');

    renderScatter(data);
    renderSilhouette(data);
    renderClusterCards(data);
  }

  function renderScatter(data) {
    const ctx = $('cluster-scatter');
    if (!ctx) return;

    const grouped = {};
    data.pca_points.forEach(p => {
      if (!grouped[p.cluster]) grouped[p.cluster] = { name: p.cluster_name, points: [] };
      grouped[p.cluster].points.push(p);
    });

    const datasets = Object.keys(grouped).sort((a, b) => +a - +b).map(cid => {
      const g = grouped[cid];
      const color = COLORS[cid % COLORS.length];
      return {
        label: 'C' + cid + ': ' + g.name.substring(0, 25),
        data: g.points.map(p => ({ x: p.x, y: p.y, label: p.label })),
        backgroundColor: color + 'cc',
        borderColor: color,
        borderWidth: 1,
        pointRadius: 4,
        pointHoverRadius: 7,
      };
    });

    if (scatterChart) scatterChart.destroy();
    scatterChart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'Class Clusters (PCA)', color: '#e2eaf4', font: { size: 14, weight: 'bold' } },
          legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 9 }, boxWidth: 10, padding: 4 } },
          tooltip: { callbacks: { label: ctx => ctx.raw.label || '' } },
        },
        scales: {
          x: { title: { display: true, text: 'PC1 (' + (data.pca_variance[0] * 100).toFixed(1) + '%)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
          y: { title: { display: true, text: 'PC2 (' + (data.pca_variance[1] * 100).toFixed(1) + '%)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
        },
      },
    });
  }

  function renderSilhouette(data) {
    const ctx = $('cluster-sil');
    if (!ctx || !data.silhouette_curve) return;
    const curve = data.silhouette_curve;

    if (silChart) silChart.destroy();
    silChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: curve.map(d => d.k),
        datasets: [{
          label: 'Silhouette',
          data: curve.map(d => d.score),
          borderColor: '#00d4ff',
          backgroundColor: 'rgba(0,212,255,0.15)',
          fill: true, borderWidth: 2.5,
          pointBackgroundColor: '#06efcf', pointBorderColor: '#fff', pointRadius: 4,
          tension: 0.3,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'Optimal k (Silhouette)', color: '#e2eaf4', font: { size: 13, weight: 'bold' } },
          legend: { display: false },
        },
        scales: {
          x: { title: { display: true, text: 'k', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
          y: { title: { display: true, text: 'Score', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
        },
      },
    });
  }

  function renderClusterCards(data) {
    const container = $('cluster-cards');
    if (!container) return;
    let html = '';
    data.clusters.forEach(cl => {
      const color = COLORS[cl.id % COLORS.length];
      const pills = cl.members.map(m => '<span class="oe-member-pill">' + m + '</span>').join('');
      html += '<div class="oe-cluster-card" style="border-left-color:' + color + '">' +
        '<div class="oe-cluster-title" style="color:' + color + '">C' + cl.id + ': ' + cl.name + '</div>' +
        '<div class="oe-cluster-size">' + cl.size + ' classes</div>' +
        '<div>' + pills + '</div></div>';
    });
    container.innerHTML = html;
  }

  // ══════════════════════════════════════════════════════════════════
  //  LLM provider single-select switches
  // ══════════════════════════════════════════════════════════════════
  const llmSwitches = Array.from(document.querySelectorAll('.oe-llm-switch'));
  llmSwitches.forEach(sw => {
    sw.addEventListener('change', function () {
      if (this.checked) {
        llmSwitches.forEach(other => { if (other !== this) other.checked = false; });
      }
    });
  });

  function getSelectedLlm() {
    const checked = llmSwitches.find(s => s.checked);
    if (!checked) return { provider: 'openai', model: '' };
    return { provider: checked.dataset.provider, model: checked.dataset.model || '' };
  }

  // ══════════════════════════════════════════════════════════════════
  //  Stage 3: Reconstruct
  // ══════════════════════════════════════════════════════════════════
  if ($('btn-reconstruct')) {
    $('btn-reconstruct').addEventListener('click', function () {
      if (!sessionId) { alert('Complete merge and cluster first.'); return; }

      const llm = getSelectedLlm();
      $('btn-reconstruct').disabled = true;
      $('reconstruct-progress').classList.remove('d-none');
      $('reconstruct-results').classList.add('d-none');
      $('recon-bar').style.width = '0%';
      $('recon-msg').textContent = 'Starting reconstruction...';

      fetch('/api/ontology-engineering/reconstruct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, provider: llm.provider, model: llm.model }),
      })
        .then(r => r.json())
        .then(data => {
          if (data.error) {
            $('btn-reconstruct').disabled = false;
            $('recon-msg').textContent = 'Error: ' + data.error;
            return;
          }
          pollReconstruct(data.task_id, data.session_id);
        })
        .catch(err => {
          $('btn-reconstruct').disabled = false;
          $('recon-msg').textContent = 'Error: ' + err.message;
        });
    });
  }

  function pollReconstruct(taskId, sessId) {
    const statusUrl = '/api/ontology-engineering/reconstruct/' + taskId + '/status';
    const interval = setInterval(() => {
      fetch(statusUrl)
        .then(r => r.json())
        .then(data => {
          const current = parseInt(data.current, 10) || 0;
          const total = parseInt(data.total, 10) || 1;
          const pct = Math.round((current / total) * 100);
          $('recon-bar').style.width = pct + '%';
          $('recon-msg').textContent = data.message || 'Processing...';

          if (data.status === 'completed') {
            clearInterval(interval);
            $('recon-bar').style.width = '100%';
            $('recon-msg').textContent = data.message || 'Reconstruction complete!';
            $('btn-reconstruct').disabled = false;
            markDone($('step-3'));
            loadReconstructedOntology(sessId);
            // Remember the reconstructed run id so View / Compare Metrics can use it.
            if (data.run_id) {
              reconstructedRunId = data.run_id;
              const viewBtn = $('btn-view-results');
              if (viewBtn) {
                viewBtn.setAttribute('data-run-id', data.run_id);
                viewBtn.classList.remove('d-none');
              }
              // Reveal Compare Metrics only if we have source runs to compare against.
              const cmpBtn = $('btn-compare-metrics');
              if (cmpBtn && sourceRunIds.length > 0) {
                cmpBtn.classList.remove('d-none');
              }
            }
          } else if (data.status === 'failed') {
            clearInterval(interval);
            $('recon-msg').textContent = 'Failed: ' + (data.error || 'Unknown error');
            $('btn-reconstruct').disabled = false;
          }
        })
        .catch(() => { /* ignore transient fetch errors */ });
    }, 800);
  }

  // ── View Full Results & Evaluation — open in an embedded iframe modal ──
  if ($('btn-view-results')) {
    $('btn-view-results').addEventListener('click', function () {
      const runId = this.getAttribute('data-run-id');
      if (!runId) { alert('No run available yet — wait for reconstruction to finish.'); return; }
      const modalEl = $('oe-full-results-modal');
      const iframe = $('oe-full-results-iframe');
      const openLink = $('oe-full-results-open');
      if (!modalEl || !iframe) return;
      iframe.src = '/results/' + encodeURIComponent(runId) + '?embedded=1';
      if (openLink) openLink.href = '/results/' + encodeURIComponent(runId);
      bootstrap.Modal.getOrCreateInstance(modalEl).show();
    });
  }

  // Clear iframe on close so nothing keeps running in the background.
  const _oeFullResultsModalEl = document.getElementById('oe-full-results-modal');
  if (_oeFullResultsModalEl) {
    _oeFullResultsModalEl.addEventListener('hidden.bs.modal', () => {
      const iframe = $('oe-full-results-iframe');
      if (iframe) iframe.src = 'about:blank';
    });
  }

  // ── Compare Metrics — reconstruction vs source run(s) ──
  if ($('btn-compare-metrics')) {
    $('btn-compare-metrics').addEventListener('click', function () {
      if (!reconstructedRunId) { alert('Reconstruction has not finished yet.'); return; }
      if (!sourceRunIds.length) { alert('No source runs were loaded in Stage 1 to compare against.'); return; }

      const modalEl = $('oe-analyze-modal');
      const body = $('oe-analyze-modal-body');
      const titleEl = $('oe-analyze-modal-title');
      if (!modalEl || !body) return;
      if (titleEl) titleEl.textContent = 'Compare metrics — Reconstruction vs source run(s)';
      body.innerHTML = '<p class="text-muted mb-0">Loading…</p>';
      bootstrap.Modal.getOrCreateInstance(modalEl).show();

      // Build label overrides so the reconstruction row stands out.
      const labelOverrides = {};
      labelOverrides[reconstructedRunId] = '★ Reconstruction (new)';
      sourceRunIds.forEach(rid => {
        const shortId = String(rid).slice(0, 8);
        labelOverrides[rid] = 'Source — ' + shortId;
      });

      const allIds = [reconstructedRunId].concat(sourceRunIds);
      fetch('/api/runs-metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_ids: allIds }),
      })
        .then(r => r.json())
        .then(data => renderAnalyzeModal(
          (data && data.rows) ? data.rows : [],
          body,
          {
            labelOverrides: labelOverrides,
            emptyMessage: 'Waiting for evaluation metrics on the reconstructed run. If no gold standard is configured, metrics cannot be computed.',
          }
        ))
        .catch(() => { body.innerHTML = '<p class="text-danger mb-0">Failed to load comparison.</p>'; });
    });
  }

  function loadReconstructedOntology(sessId) {
    fetch('/api/ontology-engineering/' + sessId + '/result')
      .then(r => r.json())
      .then(data => {
        if (data.error) {
          $('recon-msg').textContent = 'Error loading result: ' + data.error;
          return;
        }
        renderReconResults(data, sessId);
        $('reconstruct-results').classList.remove('d-none');
      })
      .catch(err => {
        $('recon-msg').textContent = 'Error: ' + err.message;
      });
  }

  function renderReconResults(data, sessId) {
    // Stats
    const s = data.stats;
    $('recon-stats').innerHTML =
      statCard(s.classes, 'Classes') +
      statCard(s.relations, 'Relations') +
      statCard(s.hierarchy, 'Hierarchy') +
      statCard(s.inferred_classes || 0, 'Inferred (new)');

    // Cytoscape graph
    renderGraph(data);

    // Download button
    if ($('btn-download-json')) {
      $('btn-download-json').onclick = function () {
        window.open('/api/ontology-engineering/' + sessId + '/result', '_blank');
      };
    }
  }

  function renderGraph(data) {
    const container = $('oe-graph-container');
    if (!container || typeof cytoscape === 'undefined') return;

    const elements = [];

    // Add nodes
    (data.nodes || []).forEach(n => {
      const d = n.data;
      const isInferred = (d.label || '').startsWith('[inferred]');
      elements.push({
        data: {
          id: d.id,
          label: (d.label || '').replace('[inferred] ', ''),
          fullLabel: d.label,
          definition: d.definition || '',
          type: d.type,
          stratum: isInferred ? 'inferred' : (d.stratum || 'core'),
        },
      });
    });

    // Add edges
    (data.edges || []).forEach(e => {
      const d = e.data;
      elements.push({
        data: {
          id: d.id,
          source: d.source,
          target: d.target,
          label: d.label || '',
          type: d.type,
        },
      });
    });

    if (cyInstance) cyInstance.destroy();
    cyInstance = cytoscape({
      container: container,
      elements: elements,
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'font-size': '10px',
            'color': '#dde9f7',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': '#22d3ee',
            'border-width': 1,
            'border-color': '#0e7490',
            'width': 30,
            'height': 30,
            'text-wrap': 'ellipsis',
            'text-max-width': '80px',
          },
        },
        {
          selector: 'node[stratum="inferred"]',
          style: {
            'background-color': '#e879f9',
            'border-color': '#a21caf',
            'shape': 'diamond',
          },
        },
        {
          selector: 'edge[type="hierarchy"]',
          style: {
            'line-color': '#34d399',
            'target-arrow-color': '#34d399',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'width': 2,
            'opacity': 0.7,
          },
        },
        {
          selector: 'edge[type="relation"]',
          style: {
            'label': 'data(label)',
            'font-size': '8px',
            'color': '#94a3b8',
            'line-color': '#a78bfa',
            'target-arrow-color': '#a78bfa',
            'target-arrow-shape': 'vee',
            'curve-style': 'bezier',
            'width': 1.5,
            'opacity': 0.6,
            'text-rotation': 'autorotate',
          },
        },
      ],
      layout: {
        name: 'cose',
        animate: false,
        nodeOverlap: 20,
        idealEdgeLength: 80,
        nodeRepulsion: 8000,
        padding: 30,
      },
      wheelSensitivity: 0.3,
    });

    // Tooltip on tap
    cyInstance.on('tap', 'node', function (evt) {
      const d = evt.target.data();
      const tip = d.fullLabel + (d.definition ? '\n' + d.definition : '');
      alert(tip);
    });
  }

})();
