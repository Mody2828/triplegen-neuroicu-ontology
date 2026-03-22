/**
 * OntologyGraph — Interactive ontology visualization for TripleGen.
 * Uses Cytoscape.js with the TripleGen glassy dark theme.
 */
const OntologyGraph = (() => {
  "use strict";

  let cy = null;
  let _overlay = null;
  let _detailEl = null;
  let _statsEl = null;
  let _activeLayout = "cose";

  /* ── Cytoscape stylesheet ────────────────────────────────────── */

  const CY_STYLE = [
    {
      selector: "node",
      style: {
        "label": "data(label)",
        "color": "#e8eef6",
        "font-size": "11px",
        "font-family": "'IBM Plex Sans', sans-serif",
        "text-halign": "center",
        "text-valign": "center",
        "text-wrap": "wrap",
        "text-max-width": "110px",
        "text-outline-color": "#060a12",
        "text-outline-width": 2,
        "background-color": "rgba(34,211,238,0.10)",
        "border-color": "#22d3ee",
        "border-width": 1.5,
        "border-opacity": 0.7,
        "shape": "roundrectangle",
        "width": "label",
        "height": "label",
        "padding": "14px",
        "transition-property": "border-color, border-width, background-color, opacity",
        "transition-duration": "0.2s",
      },
    },
    {
      selector: 'node[stratum="governance"]',
      style: {
        "background-color": "rgba(167,139,250,0.10)",
        "border-color": "#a78bfa",
      },
    },
    {
      selector: 'node[stratum="provenance"]',
      style: {
        "background-color": "rgba(251,191,36,0.10)",
        "border-color": "#fbbf24",
      },
    },
    {
      selector: 'node[stratum="inferred"]',
      style: {
        "background-color": "rgba(139,163,196,0.08)",
        "border-color": "#8ba3c4",
        "border-style": "dashed",
      },
    },
    /* Hierarchy edges — green, solid */
    {
      selector: 'edge[type="hierarchy"]',
      style: {
        "line-color": "#34d399",
        "target-arrow-color": "#34d399",
        "target-arrow-shape": "triangle",
        "arrow-scale": 1.1,
        "curve-style": "bezier",
        "width": 2,
        "opacity": 0.6,
        "transition-property": "opacity, width, line-color",
        "transition-duration": "0.2s",
      },
    },
    /* Relation edges — purple, dashed, labeled */
    {
      selector: 'edge[type="relation"]',
      style: {
        "line-color": "#a78bfa",
        "target-arrow-color": "#a78bfa",
        "target-arrow-shape": "vee",
        "arrow-scale": 1,
        "curve-style": "bezier",
        "width": 1.5,
        "opacity": 0.55,
        "line-style": "dashed",
        "line-dash-pattern": [6, 4],
        "label": "data(label)",
        "color": "#c4b5fd",
        "font-size": "9px",
        "font-family": "'IBM Plex Mono', monospace",
        "text-rotation": "autorotate",
        "text-outline-color": "#060a12",
        "text-outline-width": 2,
        "text-margin-y": -8,
        "transition-property": "opacity, width, line-color",
        "transition-duration": "0.2s",
      },
    },
    /* Interactive states */
    {
      selector: "node:selected",
      style: {
        "border-color": "#22d3ee",
        "border-width": 3,
        "background-color": "rgba(34,211,238,0.22)",
      },
    },
    {
      selector: ".og-highlight",
      style: {
        "opacity": 1,
        "border-width": 2.5,
        "z-index": 10,
      },
    },
    {
      selector: ".og-highlight-edge",
      style: {
        "opacity": 1,
        "width": 3,
        "z-index": 10,
      },
    },
    {
      selector: ".og-dim",
      style: { "opacity": 0.12 },
    },
  ];

  /* ── Layout configs ──────────────────────────────────────────── */

  function _layoutConfig(name, nodeCount) {
    const animate = nodeCount < 150;
    const configs = {
      cose: {
        name: "cose",
        animate: animate,
        animationDuration: 600,
        nodeRepulsion: () => 8000,
        idealEdgeLength: () => 120,
        edgeElasticity: () => 80,
        gravity: 0.3,
        numIter: 800,
        padding: 40,
        randomize: true,
        componentSpacing: 80,
      },
      breadthfirst: {
        name: "breadthfirst",
        directed: true,
        spacingFactor: 1.25,
        padding: 40,
        animate: animate,
        animationDuration: 500,
        avoidOverlap: true,
      },
      circle: {
        name: "circle",
        padding: 40,
        animate: animate,
        animationDuration: 400,
        avoidOverlap: true,
        spacingFactor: 1.1,
      },
      grid: {
        name: "grid",
        padding: 40,
        animate: animate,
        animationDuration: 400,
        avoidOverlap: true,
        condense: true,
      },
    };
    return configs[name] || configs.cose;
  }

  /* ── Public API ──────────────────────────────────────────────── */

  function open(runId) {
    _buildOverlay();
    _overlay.classList.add("og-visible");
    document.body.style.overflow = "hidden";
    _showLoading(true);

    fetch("/api/run/" + encodeURIComponent(runId) + "/ontology-graph")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) throw new Error(data.error);
        _render(data);
        _showLoading(false);
      })
      .catch(function (err) {
        _showLoading(false);
        _showError(err.message || "Failed to load ontology graph");
      });
  }

  function close() {
    if (cy) { cy.destroy(); cy = null; }
    if (_overlay) _overlay.classList.remove("og-visible");
    document.body.style.overflow = "";
  }

  /* ── Render ──────────────────────────────────────────────────── */

  function _render(data) {
    var container = document.getElementById("og-canvas");
    if (!container) return;

    cy = cytoscape({
      container: container,
      elements: data.nodes.concat(data.edges),
      style: CY_STYLE,
      layout: _layoutConfig(_activeLayout, data.nodes.length),
      minZoom: 0.15,
      maxZoom: 4,
      wheelSensitivity: 0.3,
    });

    _updateStats(data.stats);
    _bindEvents();
  }

  /* ── Events ──────────────────────────────────────────────────── */

  function _bindEvents() {
    cy.on("tap", "node", function (e) {
      var node = e.target;
      _highlightNeighbors(node);
      _showDetail(node);
    });

    cy.on("tap", function (e) {
      if (e.target === cy) {
        _clearHighlight();
        _hideDetail();
      }
    });

    cy.on("mouseover", "node", function (e) {
      e.target.style("cursor", "pointer");
    });
  }

  function _highlightNeighbors(node) {
    _clearHighlight();
    var neighborhood = node.neighborhood().add(node);
    cy.elements().addClass("og-dim");
    neighborhood.removeClass("og-dim");
    neighborhood.nodes().addClass("og-highlight");
    neighborhood.edges().addClass("og-highlight-edge");
    node.addClass("og-highlight");
  }

  function _clearHighlight() {
    if (!cy) return;
    cy.elements().removeClass("og-dim og-highlight og-highlight-edge");
  }

  /* ── Detail panel ────────────────────────────────────────────── */

  function _showDetail(node) {
    if (!_detailEl) return;
    var d = node.data();
    var connected = node.neighborhood("edge");
    var hierUp = connected.filter('[type="hierarchy"]').filter(function (e) { return e.data("source") === d.id; });
    var hierDown = connected.filter('[type="hierarchy"]').filter(function (e) { return e.data("target") === d.id; });
    var rels = connected.filter('[type="relation"]');

    var html = '<div class="og-detail-title">' + _esc(d.label) + "</div>";
    if (d.definition) {
      html += '<div class="og-detail-def">' + _esc(d.definition) + "</div>";
    }
    html += '<div class="og-detail-badges">';
    if (d.stratum) {
      html += '<span class="og-badge og-badge-' + _esc(d.stratum) + '">' + _esc(d.stratum) + "</span>";
    }
    html += "</div>";

    if (hierUp.length > 0) {
      html += '<div class="og-detail-section"><span class="og-detail-label">Parent(s):</span> ';
      hierUp.forEach(function (e) { html += '<span class="og-detail-tag og-tag-hier">' + _esc(e.data("target")) + "</span> "; });
      html += "</div>";
    }
    if (hierDown.length > 0) {
      html += '<div class="og-detail-section"><span class="og-detail-label">Children:</span> ';
      hierDown.forEach(function (e) { html += '<span class="og-detail-tag og-tag-hier">' + _esc(e.data("source")) + "</span> "; });
      html += "</div>";
    }
    if (rels.length > 0) {
      html += '<div class="og-detail-section"><span class="og-detail-label">Relations:</span> ';
      rels.forEach(function (e) {
        var lbl = e.data("label");
        var other = e.data("source") === d.id ? e.data("target") : e.data("source");
        var dir = e.data("source") === d.id ? " → " : " ← ";
        html += '<span class="og-detail-tag og-tag-rel">' + _esc(lbl) + dir + _esc(other) + "</span> ";
      });
      html += "</div>";
    }
    if (d.evidence) {
      html += '<div class="og-detail-evidence"><span class="og-detail-label">Evidence:</span> ' + _esc(d.evidence) + "</div>";
    }

    _detailEl.innerHTML = html;
    _detailEl.classList.add("og-detail-visible");
  }

  function _hideDetail() {
    if (_detailEl) _detailEl.classList.remove("og-detail-visible");
  }

  /* ── Stats bar ───────────────────────────────────────────────── */

  function _updateStats(stats) {
    if (!_statsEl) return;
    _statsEl.innerHTML =
      '<span class="og-stat"><i class="bi bi-circle-fill og-stat-class"></i> ' + stats.classes + " classes</span>" +
      '<span class="og-stat"><i class="bi bi-arrow-right og-stat-rel"></i> ' + stats.relations + " relations</span>" +
      '<span class="og-stat"><i class="bi bi-diagram-2 og-stat-hier"></i> ' + stats.hierarchy + " hierarchy</span>";
  }

  /* ── Layout switching ────────────────────────────────────────── */

  function _applyLayout(name) {
    if (!cy) return;
    _activeLayout = name;
    cy.layout(_layoutConfig(name, cy.nodes().length)).run();
    // Update active button
    var btns = _overlay.querySelectorAll(".og-layout-btn");
    btns.forEach(function (b) {
      b.classList.toggle("og-layout-active", b.dataset.layout === name);
    });
  }

  /* ── Export PNG ───────────────────────────────────────────────── */

  function _exportPng() {
    if (!cy) return;
    var png = cy.png({ output: "blob", bg: "#060a12", scale: 2, full: true });
    var url = URL.createObjectURL(png);
    var a = document.createElement("a");
    a.href = url;
    a.download = "ontology_graph.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  /* ── Build overlay DOM ───────────────────────────────────────── */

  function _buildOverlay() {
    if (_overlay) return;
    _overlay = document.createElement("div");
    _overlay.id = "og-overlay";
    _overlay.className = "og-overlay";
    _overlay.innerHTML =
      /* Toolbar */
      '<div class="og-toolbar">' +
        '<div class="og-toolbar-left">' +
          '<button class="og-close-btn" title="Close"><i class="bi bi-x-lg"></i></button>' +
          '<span class="og-toolbar-title"><i class="bi bi-diagram-3"></i> Ontology Graph</span>' +
          '<span id="og-stats" class="og-stats"></span>' +
        "</div>" +
        '<div class="og-toolbar-right">' +
          '<div class="og-layout-group">' +
            '<span class="og-layout-label">Layout</span>' +
            '<button class="og-layout-btn og-layout-active" data-layout="cose" title="Force-directed">Force</button>' +
            '<button class="og-layout-btn" data-layout="breadthfirst" title="Hierarchical tree">Tree</button>' +
            '<button class="og-layout-btn" data-layout="circle" title="Circular">Circle</button>' +
            '<button class="og-layout-btn" data-layout="grid" title="Grid">Grid</button>' +
          "</div>" +
          '<div class="og-action-group">' +
            '<button class="og-action-btn" id="og-btn-fit" title="Fit to screen"><i class="bi bi-arrows-fullscreen"></i></button>' +
            '<button class="og-action-btn" id="og-btn-zoomin" title="Zoom in"><i class="bi bi-zoom-in"></i></button>' +
            '<button class="og-action-btn" id="og-btn-zoomout" title="Zoom out"><i class="bi bi-zoom-out"></i></button>' +
            '<button class="og-action-btn" id="og-btn-png" title="Export PNG"><i class="bi bi-download"></i> PNG</button>' +
          "</div>" +
        "</div>" +
      "</div>" +
      /* Canvas */
      '<div id="og-canvas" class="og-canvas"></div>' +
      /* Legend */
      '<div class="og-legend">' +
        '<div class="og-legend-title">Legend</div>' +
        '<div class="og-legend-item"><span class="og-legend-swatch og-swatch-class"></span> Class (core)</div>' +
        '<div class="og-legend-item"><span class="og-legend-swatch og-swatch-gov"></span> Class (governance)</div>' +
        '<div class="og-legend-item"><span class="og-legend-swatch og-swatch-prov"></span> Class (provenance)</div>' +
        '<div class="og-legend-item"><span class="og-legend-line og-line-hier"></span> subClassOf</div>' +
        '<div class="og-legend-item"><span class="og-legend-line og-line-rel"></span> Relation</div>' +
      "</div>" +
      /* Detail panel */
      '<div id="og-detail" class="og-detail"></div>' +
      /* Loading spinner */
      '<div id="og-loading" class="og-loading">' +
        '<div class="spinner-border text-info" role="status"><span class="visually-hidden">Loading...</span></div>' +
        '<div class="og-loading-text">Loading ontology graph...</div>' +
      "</div>" +
      /* Error display */
      '<div id="og-error" class="og-error-msg"></div>';

    document.body.appendChild(_overlay);
    _detailEl = document.getElementById("og-detail");
    _statsEl = document.getElementById("og-stats");

    // Bind toolbar events
    _overlay.querySelector(".og-close-btn").addEventListener("click", close);
    _overlay.querySelectorAll(".og-layout-btn").forEach(function (btn) {
      btn.addEventListener("click", function () { _applyLayout(btn.dataset.layout); });
    });
    document.getElementById("og-btn-fit").addEventListener("click", function () { if (cy) cy.fit(undefined, 40); });
    document.getElementById("og-btn-zoomin").addEventListener("click", function () { if (cy) cy.zoom({ level: cy.zoom() * 1.3, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }); });
    document.getElementById("og-btn-zoomout").addEventListener("click", function () { if (cy) cy.zoom({ level: cy.zoom() / 1.3, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }); });
    document.getElementById("og-btn-png").addEventListener("click", _exportPng);

    // ESC to close
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && _overlay.classList.contains("og-visible")) close();
    });
  }

  /* ── Helpers ─────────────────────────────────────────────────── */

  function _showLoading(show) {
    var el = document.getElementById("og-loading");
    if (el) el.style.display = show ? "flex" : "none";
  }

  function _showError(msg) {
    var el = document.getElementById("og-error");
    if (el) { el.textContent = msg; el.style.display = "block"; }
  }

  function _esc(s) {
    if (!s) return "";
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  return { open: open, close: close };
})();
