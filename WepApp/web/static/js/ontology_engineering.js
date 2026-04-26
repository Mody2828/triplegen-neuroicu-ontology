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

  // ── Unified "Load previous work" dropdown ──
  // Three optgroups, in priority order (fullest data first):
  //   Reconstructions  — cascade Stages 1 → 4
  //   Clusterings      — cascade Stages 1 → 2
  //   Pipeline Runs    — cascade Stage 4 only
  // Called on page load AND after each successful cluster/reconstruct, so a
  // newly-saved session appears without a full reload.
  function populateUnifiedLoader() {
    const sel = $('oe-unified-load');
    if (!sel) return Promise.resolve();
    const prev = sel.value;
    while (sel.options.length > 1) sel.remove(1);

    const fetchJson = (url) => fetch(url)
      .then(r => r.ok ? r.json() : [])
      .catch(() => []);

    return Promise.all([
      fetchJson('/api/ontology-engineering/sessions'),
      fetchJson('/api/ontology-engineering/cluster-sessions'),
      fetchJson('/api/ontology-engineering/judgeable-runs'),
    ]).then(([reconstructions, clusterings, runs]) => {
      const reconIds = new Set((reconstructions || []).map(s => s.session_id));

      if (reconstructions && reconstructions.length) {
        const g = document.createElement('optgroup');
        g.label = 'Reconstructions (Stages 1–4)';
        reconstructions.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.session_id;
          opt.dataset.kind = 'reconstruction';
          opt.dataset.runId = s.run_id || '';
          opt.dataset.sourceRuns = JSON.stringify(s.source_runs || []);
          opt.textContent = s.session_id.replace('oe-', '') + ' — ' + s.n_classes + ' classes';
          g.appendChild(opt);
        });
        sel.appendChild(g);
      }

      // Clusterings that don't also have a reconstruction — otherwise the
      // recon version (richer) already covers them.
      const clusterOnly = (clusterings || []).filter(s => !reconIds.has(s.session_id));
      if (clusterOnly.length) {
        const g = document.createElement('optgroup');
        g.label = 'Clusterings only (Stages 1–2)';
        clusterOnly.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.session_id;
          opt.dataset.kind = 'clustering';
          opt.dataset.sourceRuns = JSON.stringify(s.source_runs || []);
          opt.textContent = s.session_id.replace('oe-', '') +
            ' — ' + s.n_clusters + ' clusters, ' + s.total_classes + ' classes';
          g.appendChild(opt);
        });
        sel.appendChild(g);
      }

      if (runs && runs.length) {
        const g = document.createElement('optgroup');
        g.label = 'Pipeline Runs (Stage 4 only)';
        runs.forEach(r => {
          const opt = document.createElement('option');
          opt.value = r.run_id;
          opt.dataset.kind = 'run';
          const marker = r.has_verdicts ? ' ✓' : '';
          const paper = r.paper_name ? ' — ' + r.paper_name : '';
          opt.textContent = r.run_id + paper +
            ' — ' + r.n_triples + ' triples' + marker;
          g.appendChild(opt);
        });
        sel.appendChild(g);
      }

      if (prev) {
        for (let i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === prev) { sel.selectedIndex = i; break; }
        }
      }
    });
  }


  function unlock(stageEl, stepEl) {
    stageEl.classList.remove('locked');
    stepEl.classList.add('active');
  }

  function markDone(stepEl) {
    stepEl.classList.remove('active');
    stepEl.classList.add('done');
  }

  function statCard(value, label, accent) {
    // `accent` (optional) is a hex colour used to tint the card:
    //   - 4px coloured left border
    //   - value text takes the accent colour (overrides default)
    const cardStyle = accent
      ? ` style="border-left: 4px solid ${accent};"`
      : '';
    const valueStyle = accent
      ? ` style="color: ${accent};"`
      : '';
    return `<div class="oe-stat-card"${cardStyle}>
      <div class="oe-stat-value"${valueStyle}>${value}</div>
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
    $('btn-select-all').addEventListener('click', () => { checkboxes().forEach(c => { if (c.closest('.oe-run-item').style.display !== 'none') c.checked = true; }); updateCount(); });
  }
  if ($('btn-deselect-all')) {
    $('btn-deselect-all').addEventListener('click', () => { checkboxes().forEach(c => c.checked = false); updateCount(); });
  }
  document.querySelectorAll('.oe-run-check').forEach(c => c.addEventListener('change', updateCount));
  updateCount();

  // ── Pipeline mode filter ──
  document.querySelectorAll('input[name="oe-mode-filter"]').forEach(radio => {
    radio.addEventListener('change', () => {
      const mode = radio.value;
      document.querySelectorAll('.oe-run-item').forEach(item => {
        if (mode === 'all' || item.dataset.mode === mode) {
          item.style.display = '';
        } else {
          item.style.display = 'none';
          // uncheck hidden items so they don't get included
          const cb = item.querySelector('.oe-run-check');
          if (cb) cb.checked = false;
        }
      });
      updateCount();
    });
  });

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

          // Refresh the unified loader so the newly-saved cluster session
          // appears immediately without a page reload.
          populateUnifiedLoader();
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

  // Judge provider switches — independent of reconstruction provider so users
  // can run the qualitative judge on a pre-existing session with a different
  // model (e.g. GPT-4o for judging even when reconstruction used GPT-4o-mini).
  const qevalLlmSwitches = Array.from(document.querySelectorAll('.oe-qeval-llm-switch'));
  qevalLlmSwitches.forEach(sw => {
    sw.addEventListener('change', function () {
      if (this.checked) {
        qevalLlmSwitches.forEach(other => { if (other !== this) other.checked = false; });
      }
    });
  });

  function getSelectedQevalLlm() {
    const checked = qevalLlmSwitches.find(s => s.checked);
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
            showQualitativeEvalSection(sessId);

            // Refresh the unified loader so the newly-saved reconstruction
            // appears immediately without a page reload.
            populateUnifiedLoader();
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

    // Download buttons
    if ($('btn-download-json')) {
      $('btn-download-json').onclick = function () {
        window.open('/api/ontology-engineering/' + sessId + '/result', '_blank');
      };
    }
    if ($('btn-download-ttl')) {
      $('btn-download-ttl').onclick = function () {
        window.open('/api/ontology-engineering/' + sessId + '/ttl', '_blank');
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

  // ══════════════════════════════════════════════════════════════════
  //  Unified "Load previous work" — single entry point for everything below
  // ══════════════════════════════════════════════════════════════════
  const unifiedLoader = $('oe-unified-load');
  if (unifiedLoader) {
    populateUnifiedLoader().then(() => {
      // Honour ?load=<id> in the URL (used by the "Load cleaned session"
      // link after applying judge edits, and by legacy deep-links).
      try {
        const params = new URLSearchParams(window.location.search);
        const wantLoad = params.get('load');
        if (wantLoad) {
          for (let i = 0; i < unifiedLoader.options.length; i++) {
            if (unifiedLoader.options[i].value === wantLoad) {
              unifiedLoader.selectedIndex = i;
              unifiedLoader.dispatchEvent(new Event('change'));
              break;
            }
          }
        }
      } catch (_) { /* ignore */ }
    });

    unifiedLoader.addEventListener('change', function () {
      const id = this.value;
      const hint = $('oe-unified-load-hint');
      if (!id) { if (hint) hint.textContent = ''; return; }
      const opt = this.options[this.selectedIndex];
      const kind = opt.dataset.kind || '';
      if (hint) hint.textContent = '· loaded as ' + kind;

      if (kind === 'reconstruction') {
        cascadeReconstruction(id, opt.dataset.runId || '', JSON.parse(opt.dataset.sourceRuns || '[]'));
      } else if (kind === 'clustering') {
        cascadeClustering(id, JSON.parse(opt.dataset.sourceRuns || '[]'));
      } else if (kind === 'run') {
        cascadeRegularRun(id);
      }
    });
  }

  // ── Cascade helpers ─────────────────────────────────────────────────
  // Each helper is a pure function of (id, side-data) → DOM effects.
  // Kept separate so the unified loader stays a thin dispatcher.

  function cascadeReconstruction(sessId, runId, srcRuns) {
    // Set global OE state as if we had just finished the full pipeline.
    sessionId = sessId;
    sourceRunIds = srcRuns || [];
    reconstructedRunId = runId || null;

    // Stage 2 — unlock + mark done (reconstruction implies clustering was done)
    const stage2 = $('stage-cluster');
    if (stage2) stage2.classList.remove('locked');
    markDone($('step-2'));
    // Best-effort cluster-data fetch (reconstruction sessions keep it on disk)
    fetch('/api/ontology-engineering/' + sessId + '/cluster-data')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && !data.error) {
          renderClusterResults(data);
          $('cluster-results').classList.remove('d-none');
        }
      })
      .catch(() => {});

    // Stage 3 — unlock + mark done
    const stage3 = $('stage-reconstruct');
    if (stage3) stage3.classList.remove('locked');
    markDone($('step-3'));
    $('reconstruct-progress').classList.add('d-none');
    $('reconstruct-results').classList.remove('d-none');

    const viewBtn = $('btn-view-results');
    if (viewBtn && runId) {
      viewBtn.setAttribute('data-run-id', runId);
      viewBtn.classList.remove('d-none');
    }
    const cmpBtn = $('btn-compare-metrics');
    if (cmpBtn && srcRuns && srcRuns.length > 0 && runId) {
      cmpBtn.classList.remove('d-none');
    }

    // Load the reconstructed graph + expose Stage 4 (qeval auto-renders if saved)
    loadReconstructedOntology(sessId);
    showQualitativeEvalSection(sessId);
  }

  function cascadeClustering(sessId, srcRuns) {
    sessionId = sessId;
    sourceRunIds = srcRuns || [];
    reconstructedRunId = null;

    const stage2 = $('stage-cluster');
    if (stage2) stage2.classList.remove('locked');
    markDone($('step-2'));
    unlock($('stage-reconstruct'), $('step-3'));

    fetch('/api/ontology-engineering/' + sessId + '/cluster-data')
      .then(r => r.json())
      .then(data => {
        if (data.error) { alert('Error loading cluster: ' + data.error); return; }
        renderClusterResults(data);
        $('cluster-results').classList.remove('d-none');
      })
      .catch(err => alert('Error: ' + err.message));
    // Stage 4 stays locked — no ontology to judge until reconstruction runs.
  }

  function cascadeRegularRun(runId) {
    // No merge / clustering / reconstruction — Stage 4 direct entry.
    // Stages 1-3 stay in their current (typically locked) state so the user
    // sees the card greyed out, which correctly signals "nothing to populate".
    showQualitativeEvalSection(runId);
    const step4 = $('step-4');
    if (step4 && !step4.classList.contains('done')) step4.classList.add('active');
  }

  // ══════════════════════════════════════════════════════════════════
  //  Qualitative LLM-as-judge evaluation
  // ══════════════════════════════════════════════════════════════════
  const QEVAL_VERDICT_COLORS = {
    accept_direct: '#2ED6A1',
    accept_indirect: '#6CC2FF',
    revise: '#F5A65B',
    reject: '#FF6B81',
  };
  const QEVAL_VERDICT_LABELS = {
    accept_direct: 'Accept — direct',
    accept_indirect: 'Accept — indirect',
    revise: 'Revise',
    reject: 'Reject',
  };
  const QEVAL_SUBTYPE_LABELS = {
    '4.1': 'Target more general',
    '4.2': 'Target more specific',
    '4.3': 'Target different class',
    '4.4': 'Replace with direct relation',
    '4.5': 'Replace with indirect relation',
    '4.6': 'Edit class name',
    '4.7': 'Subclass → relation',
    '4.8': 'Swap subject/object',
    '4.9': 'Relation → subclass',
    '4.10': 'Other revise',
  };

  // Approximate USD per 1M tokens (April 2026 published rates). Keyed by
  // provider; models override via optional modelKey match on data-model.
  // Numbers are intentionally round — the preflight line is an estimate.
  const QEVAL_PRICING = {
    openai: {                                // default: gpt-4o-mini
      input: 0.15, output: 0.60,
      models: {
        'gpt-4o': { input: 2.50, output: 10.00 },
      },
    },
    anthropic: { input: 1.00, output: 5.00 },   // Claude Haiku 4.5
    google:    { input: 0.15, output: 0.60 },   // Gemini 2.5 Flash
    groq:      { input: 0.00, output: 0.00 },   // free tier
    huggingface: { input: 0.05, output: 0.10 },
    deepseek:  { input: 0.14, output: 0.28 },
  };

  // Rough per-call token accounting derived from the actual prompt templates.
  // Stage A fires on every triple; Stage B fires on ~30% (revise rate).
  const QEVAL_TOKENS = {
    stageA: { input: 550, output: 200 },
    stageB: { input: 600, output: 250 },
    reviseRate: 0.30,
  };

  function estimateJudgeCost(nTriples, provider, model) {
    const row = QEVAL_PRICING[provider];
    if (!row) return null;
    const rate = (model && row.models && row.models[model]) ? row.models[model] : row;
    const t = QEVAL_TOKENS;
    const inTok  = nTriples * (t.stageA.input  + t.reviseRate * t.stageB.input);
    const outTok = nTriples * (t.stageA.output + t.reviseRate * t.stageB.output);
    const usd = (inTok * rate.input + outTok * rate.output) / 1_000_000;
    return usd;
  }

  function formatCost(usd) {
    if (usd == null) return '?';
    if (usd === 0) return 'free';
    if (usd < 0.01) return '<$0.01';
    if (usd < 1) return '$' + usd.toFixed(2);
    return '$' + usd.toFixed(2);
  }

  function updateQevalPreflight() {
    const countEl = $('qeval-triple-count');
    const section = $('qualitative-eval-section');
    if (!countEl || !section) return;
    const n = parseInt(section.dataset.tripleCount, 10);
    if (!n) { countEl.textContent = ''; return; }
    const llm = getSelectedQevalLlm();
    const providerLabel = llm.model ? (llm.provider + ' / ' + llm.model) : llm.provider;
    const cost = estimateJudgeCost(n, llm.provider, llm.model);
    countEl.textContent = '≈ ' + n + ' triples · judge: ' + providerLabel +
      ' · est. ' + formatCost(cost);
  }

  // Re-estimate cost whenever the user switches the judge provider.
  qevalLlmSwitches.forEach(sw => sw.addEventListener('change', updateQevalPreflight));

  function showQualitativeEvalSection(sessId) {
    const section = $('qualitative-eval-section');
    if (!section || !sessId) return;
    section.classList.remove('d-none');
    // Unlock the Stage-4 card and advance the stepper. The card is visually
    // greyed out by default until a session is loaded.
    const stage4 = $('stage-qualitative-eval');
    if (stage4) stage4.classList.remove('locked');
    const step4 = $('step-4');
    if (step4 && !step4.classList.contains('done')) step4.classList.add('active');

    const btn = $('btn-qualitative-eval');
    const btnText = $('btn-qualitative-eval-text');
    if (btn) btn.disabled = false;
    if (btnText) btnText.textContent = 'Run LLM Qualitative Evaluation';
    $('qeval-progress').classList.add('d-none');
    $('qeval-results').classList.add('d-none');
    section.dataset.sessionId = sessId;

    // OE sessions use the OE result endpoint; regular pipeline runs use the
    // per-run ontology-graph endpoint. Both return a {stats:{relations,hierarchy}}
    // payload we can count triples from.
    const isOe = /^oe-/.test(sessId);
    const statsUrl = isOe
      ? '/api/ontology-engineering/' + sessId + '/result'
      : '/api/run/' + sessId + '/ontology-graph';

    // Apply-edits produces a <id>-cleaned OE session — only meaningful for
    // OE reconstructions. Hide the whole block for regular runs.
    const applyBlock = $('qeval-apply-block');
    if (applyBlock) applyBlock.style.display = isOe ? '' : 'none';

    // Preflight: load triple count from the loaded ontology, cache on the
    // section element, then render the preflight line (count + cost).
    fetch(statsUrl)
      .then(r => r.json())
      .then(data => {
        if (data && data.stats) {
          const n = (data.stats.relations || 0) + (data.stats.hierarchy || 0);
          section.dataset.tripleCount = String(n);
          updateQevalPreflight();
        }
      })
      .catch(() => {});

    // Load any prior eval results for this session. If present, render them
    // immediately and flip the button label to "Re-run" so the user knows
    // clicking will overwrite the saved verdicts.
    fetch('/api/ontology-engineering/' + sessId + '/qualitative-eval')
      .then(r => r.ok ? r.json() : null)
      .then(payload => {
        if (payload && payload.summary) {
          renderQualitativeResults(sessId, payload.summary, payload.verdicts || []);
          if (btnText) btnText.textContent = 'Re-run LLM Qualitative Evaluation';
          markDone(step4);
        }
      })
      .catch(() => {});
  }

  function renderQualitativeResults(sessId, summary, verdicts) {
    const resultsEl = $('qeval-results');
    if (!resultsEl) return;
    resultsEl.classList.remove('d-none');

    // Stat cards
    const statsEl = $('qeval-stats');
    if (statsEl) {
      const counts = summary.verdict_counts || {};
      const pct = summary.verdict_pct || {};
      const pctStr = (k) => (pct[k] != null ? pct[k].toFixed(1) + '%' : '—');
      statsEl.innerHTML =
        statCard(summary.n_triples || 0, 'Triples judged') +
        statCard((counts.accept_direct || 0) + ' <small class="text-muted">(' + pctStr('accept_direct') + ')</small>', 'Accept — direct', QEVAL_VERDICT_COLORS.accept_direct) +
        statCard((counts.accept_indirect || 0) + ' <small class="text-muted">(' + pctStr('accept_indirect') + ')</small>', 'Accept — indirect', QEVAL_VERDICT_COLORS.accept_indirect) +
        statCard((counts.revise || 0) + ' <small class="text-muted">(' + pctStr('revise') + ')</small>', 'Revise', QEVAL_VERDICT_COLORS.revise) +
        statCard((counts.reject || 0) + ' <small class="text-muted">(' + pctStr('reject') + ')</small>', 'Reject', QEVAL_VERDICT_COLORS.reject) +
        statCard((summary.action_weighted_quality != null ? (summary.action_weighted_quality * 100).toFixed(1) + '%' : '—'), 'Action-weighted quality') +
        statCard(summary.engineer_effort_count || 0, 'KE-effort triples') +
        statCard((summary.mean_confidence != null ? summary.mean_confidence.toFixed(2) : '—'), 'Mean confidence');
    }

    // Revise breakdown bars
    const breakdownWrap = $('qeval-breakdown-wrap');
    const breakdownEl = $('qeval-breakdown');
    const reviseCounts = summary.revise_breakdown_counts || {};
    const reviseTotal = Object.values(reviseCounts).reduce((a, b) => a + (b || 0), 0);
    if (breakdownWrap && breakdownEl) {
      if (reviseTotal > 0) {
        let html = '';
        Object.keys(QEVAL_SUBTYPE_LABELS).forEach(k => {
          const c = reviseCounts[k] || 0;
          if (c === 0) return;
          const w = Math.round((c / reviseTotal) * 100);
          html += '<div class="mb-1"><div class="d-flex justify-content-between">' +
            '<span><strong>' + k + '</strong> — ' + QEVAL_SUBTYPE_LABELS[k] + '</span>' +
            '<span class="text-muted">' + c + '</span></div>' +
            '<div style="height:6px; background:var(--tg-border); border-radius:3px; overflow:hidden;">' +
            '<div style="height:100%; width:' + w + '%; background:' + QEVAL_VERDICT_COLORS.revise + ';"></div>' +
            '</div></div>';
        });
        breakdownEl.innerHTML = html;
        breakdownWrap.style.display = '';
      } else {
        breakdownWrap.style.display = 'none';
      }
    }

    // Per-triple table (populated but hidden by default)
    const tbody = $('qeval-tbody');
    if (tbody) {
      tbody.innerHTML = verdicts.map((v, i) => {
        const color = QEVAL_VERDICT_COLORS[v.verdict] || '#999';
        let verdictCell = '<span style="color:' + color + '; font-weight:600;">' +
          (QEVAL_VERDICT_LABELS[v.verdict] || v.verdict) + '</span>';
        if (v.revise_subtype) {
          verdictCell += ' <small class="text-muted">(' + v.revise_subtype + ')</small>';
        }
        const sug = v.suggested_triple;
        const sugText = (sug && sug.subject)
          ? (sug.subject + ' — ' + sug.relation + ' — ' + sug.object)
          : '—';
        const conf = (v.confidence != null) ? v.confidence.toFixed(2) : '—';
        // Only reject/revise rows are meaningful edits — accept rows are
        // informational, so we disable their checkboxes.
        const editable = (v.verdict === 'reject' || v.verdict === 'revise');
        const tIdx = (v.triple_index != null) ? v.triple_index : i;
        const checkbox = editable
          ? '<input type="checkbox" class="qeval-row-check" data-triple-index="' + tIdx +
            '" data-verdict="' + escapeHtml(v.verdict) + '">'
          : '<span class="text-muted" title="Accepted — no edit needed">–</span>';
        return '<tr data-verdict="' + escapeHtml(v.verdict) + '">' +
          '<td>' + checkbox + '</td>' +
          '<td>' + (i + 1) + '</td>' +
          '<td><strong>' + escapeHtml(v.subject) + '</strong> — <em>' +
            escapeHtml(v.relation) + '</em> — <strong>' + escapeHtml(v.object) + '</strong>' +
            '<div class="text-muted" style="font-size:0.7rem;">' + escapeHtml(v.justification || '') + '</div></td>' +
          '<td>' + verdictCell + '</td>' +
          '<td>' + escapeHtml(sugText) + '</td>' +
          '<td>' + conf + '</td>' +
        '</tr>';
      }).join('');
      updateQevalSelectCount();
    }

    // CSV link
    const csvBtn = $('btn-qeval-csv');
    if (csvBtn) csvBtn.href = '/api/ontology-engineering/' + sessId + '/qualitative-eval.csv';

    // Reset apply-edits panel when new verdicts load.
    const applyResult = $('qeval-apply-result');
    if (applyResult) { applyResult.classList.add('d-none'); applyResult.innerHTML = ''; }
    const applyBtn = $('btn-qeval-apply');
    if (applyBtn) applyBtn.disabled = true;
  }

  // ── Checkbox / selection helpers ────────────────────────────────────

  function getCheckedTripleIndexes() {
    return Array.from(document.querySelectorAll('.qeval-row-check:checked'))
      .map(cb => parseInt(cb.getAttribute('data-triple-index'), 10))
      .filter(n => !isNaN(n));
  }

  function updateQevalSelectCount() {
    const countEl = $('qeval-select-count');
    const applyBtn = $('btn-qeval-apply');
    const n = getCheckedTripleIndexes().length;
    if (countEl) countEl.textContent = n ? (n + ' selected') : '';
    if (applyBtn) applyBtn.disabled = (n === 0);
  }

  // Delegate checkbox changes from the table body.
  document.addEventListener('change', function (e) {
    if (e.target && e.target.classList && e.target.classList.contains('qeval-row-check')) {
      updateQevalSelectCount();
    }
  });

  // Header "check all visible" toggles every row checkbox.
  if ($('qeval-check-all')) {
    $('qeval-check-all').addEventListener('change', function () {
      const want = this.checked;
      document.querySelectorAll('.qeval-row-check').forEach(cb => { cb.checked = want; });
      updateQevalSelectCount();
    });
  }

  // Quick-select buttons (rejects / revises / all-edits / none).
  document.querySelectorAll('[data-qeval-select]').forEach(btn => {
    btn.addEventListener('click', function () {
      const mode = this.getAttribute('data-qeval-select');
      document.querySelectorAll('.qeval-row-check').forEach(cb => {
        const v = cb.getAttribute('data-verdict');
        let want = false;
        if (mode === 'rejects')   want = (v === 'reject');
        else if (mode === 'revises') want = (v === 'revise');
        else if (mode === 'all-edits') want = (v === 'reject' || v === 'revise');
        else if (mode === 'none') want = false;
        cb.checked = want;
      });
      const headerCb = $('qeval-check-all');
      if (headerCb) headerCb.checked = false;
      updateQevalSelectCount();
    });
  });

  // ── Apply judge edits ───────────────────────────────────────────────

  if ($('btn-qeval-apply')) {
    $('btn-qeval-apply').addEventListener('click', function () {
      const section = $('qualitative-eval-section');
      const sessId = section && section.dataset.sessionId;
      if (!sessId) { alert('No OE session loaded.'); return; }
      const approved = getCheckedTripleIndexes();
      if (!approved.length) {
        alert('Check at least one triple to apply judge edits.');
        return;
      }

      const btn = this;
      const resultEl = $('qeval-apply-result');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Applying...';
      if (resultEl) { resultEl.classList.add('d-none'); resultEl.innerHTML = ''; }

      fetch('/api/ontology-engineering/' + sessId + '/apply-judge-edits', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved_indexes: approved }),
      })
        .then(r => r.json().then(data => ({ ok: r.ok, data })))
        .then(({ ok, data }) => {
          if (!ok || data.error) {
            if (resultEl) {
              resultEl.classList.remove('d-none');
              resultEl.innerHTML = '<div class="text-danger">Error: ' +
                escapeHtml(data.error || 'Unknown error') + '</div>';
            }
            return;
          }
          renderApplyResult(sessId, data);
        })
        .catch(err => {
          if (resultEl) {
            resultEl.classList.remove('d-none');
            resultEl.innerHTML = '<div class="text-danger">Error: ' + escapeHtml(err.message) + '</div>';
          }
        })
        .finally(() => {
          btn.innerHTML = '<i class="bi bi-check2-square me-1"></i>Apply Judge Edits to Approved Triples';
          updateQevalSelectCount();
        });
    });
  }

  function renderApplyResult(sessId, stats) {
    const el = $('qeval-apply-result');
    if (!el) return;
    el.classList.remove('d-none');
    const inRel = stats.input_relations || 0;
    const inHier = stats.input_hierarchy || 0;
    const outRel = stats.output_relations != null ? stats.output_relations : '?';
    const outHier = stats.output_hierarchy != null ? stats.output_hierarchy : '?';
    const totIn = inRel + inHier;
    const totOut = (typeof outRel === 'number' && typeof outHier === 'number') ? (outRel + outHier) : '?';
    const subApplied = stats.by_subtype_applied || {};
    const subSkipped = stats.by_subtype_skipped || {};
    const subLine = (obj) => Object.keys(obj).sort().map(k => k + '×' + obj[k]).join(', ') || '—';
    const newSessId = stats.new_session_id || '';
    const downloadUrl = stats.download_url ||
      ('/api/ontology-engineering/' + (newSessId || sessId) + '/reconstructed-ontology');

    const newSessBlock = newSessId
      ? '<div class="mb-2 p-2" style="background:rgba(46,214,161,0.08); border:1px solid var(--tg-accent-1); border-radius:var(--tg-radius-sm);">' +
        '  <div class="small text-muted mb-1">New cleaned session saved as</div>' +
        '  <div style="font-family:monospace; font-size:0.85rem; color:var(--tg-accent-1); word-break:break-all;">' +
             escapeHtml(newSessId) + '</div>' +
        '  <div class="small text-muted mt-1">Appears in the <em>Load previous OE run</em> dropdown at the top of this page.</div>' +
        '</div>'
      : '';

    const loadBtn = newSessId
      ? '    <button type="button" class="btn btn-tg-secondary btn-sm ms-2" id="btn-qeval-load-cleaned" data-sess-id="' +
          escapeHtml(newSessId) + '">' +
        '      <i class="bi bi-arrow-repeat me-1"></i>Load cleaned session' +
        '    </button>'
      : '';

    el.innerHTML =
      '<div class="p-2" style="background:var(--tg-surface); border:1px solid var(--tg-border); border-radius:var(--tg-radius-sm);">' +
      '  <div class="fw-semibold mb-2" style="color:var(--tg-accent-1);">' +
      '    <i class="bi bi-check-circle me-1"></i>Cleaned ontology written to new session' +
      '  </div>' +
      newSessBlock +
      '  <div class="row" style="font-size:0.78rem;">' +
      '    <div class="col-md-6">' +
      '      <div><strong>Approved:</strong> ' + (stats.approved_count || 0) +
             ' / ' + (stats.total_verdicts || 0) + ' verdicts</div>' +
      '      <div><strong>Dropped (reject):</strong> ' + (stats.dropped_reject || 0) + '</div>' +
      '      <div><strong>Revises applied:</strong> ' + (stats.revise_applied || 0) +
             ' <small class="text-muted">(' + subLine(subApplied) + ')</small></div>' +
      '      <div><strong>Revises skipped (new class):</strong> ' + (stats.revise_skipped_new_class || 0) +
             ' <small class="text-muted">(' + subLine(subSkipped) + ')</small></div>' +
      '    </div>' +
      '    <div class="col-md-6">' +
      '      <div><strong>Relations:</strong> ' + inRel + ' → ' + outRel + '</div>' +
      '      <div><strong>Hierarchy:</strong> ' + inHier + ' → ' + outHier + '</div>' +
      '      <div><strong>Total triples:</strong> ' + totIn + ' → ' + totOut + '</div>' +
      '    </div>' +
      '  </div>' +
      '  <div class="mt-2">' +
      '    <a class="btn btn-tg-secondary btn-sm" href="' + escapeHtml(downloadUrl) + '" download>' +
      '      <i class="bi bi-download me-1"></i>Download cleaned ontology (JSON)' +
      '    </a>' +
           loadBtn +
      '  </div>' +
      '</div>';

    // Wire the "Load cleaned session" button to reload the page with the new
    // session pre-loaded via the existing "Load previous OE run" dropdown.
    const loadBtnEl = $('btn-qeval-load-cleaned');
    if (loadBtnEl) {
      loadBtnEl.addEventListener('click', function () {
        const nsid = this.getAttribute('data-sess-id');
        if (!nsid) return;
        // Reload the OE page; user can pick the new session from the dropdown
        // (it's listed automatically since it has reconstructed_ontology.json).
        window.location.href = '/ontology-engineering?load=' + encodeURIComponent(nsid);
      });
    }
  }

  function escapeHtml(s) {
    if (s == null) return '';
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  if ($('btn-qualitative-eval')) {
    $('btn-qualitative-eval').addEventListener('click', function () {
      const section = $('qualitative-eval-section');
      const sessId = section && section.dataset.sessionId;
      if (!sessId) { alert('Load a run or OE session first.'); return; }

      const llm = getSelectedQevalLlm();
      this.disabled = true;
      $('qeval-progress').classList.remove('d-none');
      $('qeval-results').classList.add('d-none');
      $('qeval-bar').style.width = '0%';
      $('qeval-msg').textContent = 'Starting qualitative eval...';

      fetch('/api/ontology-engineering/qualitative-eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessId, provider: llm.provider, model: llm.model }),
      })
        .then(r => r.json())
        .then(data => {
          if (data.error) {
            $('qeval-msg').textContent = 'Error: ' + data.error;
            $('btn-qualitative-eval').disabled = false;
            return;
          }
          pollQualitativeEval(data.task_id, sessId);
        })
        .catch(err => {
          $('qeval-msg').textContent = 'Error: ' + err.message;
          $('btn-qualitative-eval').disabled = false;
        });
    });
  }

  if ($('btn-qeval-toggle-table')) {
    $('btn-qeval-toggle-table').addEventListener('click', function () {
      const wrap = $('qeval-table-wrap');
      if (!wrap) return;
      const hidden = wrap.classList.toggle('d-none');
      this.innerHTML = hidden
        ? '<i class="bi bi-table me-1"></i>Show per-triple verdicts'
        : '<i class="bi bi-chevron-up me-1"></i>Hide per-triple verdicts';
    });
  }

  function pollQualitativeEval(taskId, sessId) {
    const statusUrl = '/api/ontology-engineering/qualitative-eval/' + taskId + '/status';
    const interval = setInterval(() => {
      fetch(statusUrl)
        .then(r => r.json())
        .then(data => {
          const current = parseInt(data.current, 10) || 0;
          const total = parseInt(data.total, 10) || 1;
          const pct = Math.round((current / total) * 100);
          $('qeval-bar').style.width = pct + '%';
          $('qeval-msg').textContent = data.message || 'Judging...';

          if (data.status === 'completed') {
            clearInterval(interval);
            $('qeval-bar').style.width = '100%';
            $('qeval-msg').textContent = data.message || 'Qualitative eval complete.';
            $('btn-qualitative-eval').disabled = false;
            const btnText = $('btn-qualitative-eval-text');
            if (btnText) btnText.textContent = 'Re-run LLM Qualitative Evaluation';
            markDone($('step-4'));
            // Fetch final results
            fetch('/api/ontology-engineering/' + sessId + '/qualitative-eval')
              .then(r => r.json())
              .then(payload => {
                if (payload && payload.summary) {
                  renderQualitativeResults(sessId, payload.summary, payload.verdicts || []);
                }
              })
              .catch(() => {});
          } else if (data.status === 'failed') {
            clearInterval(interval);
            $('qeval-msg').textContent = 'Failed: ' + (data.error || 'Unknown error');
            $('btn-qualitative-eval').disabled = false;
          }
        })
        .catch(() => { /* ignore transient */ });
    }, 800);
  }

})();
