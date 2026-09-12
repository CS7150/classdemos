// Keep the figure under half a screen, then report our height so the iframe fits it exactly.
(function () {
  var MAX = 450, last = 0;

  function svgs() { return document.querySelectorAll('svg[viewBox]'); }
  function height() { return Math.ceil(document.documentElement.getBoundingClientRect().height); }

  // scale each drawing by width rather than letterboxing it, so the cap costs no side space
  function fit(cap) {
    var v = svgs();
    for (var i = 0; i < v.length; i++) {
      var b = v[i].viewBox.baseVal;
      if (b && b.height) v[i].style.maxWidth = Math.round(cap * b.width / b.height) + 'px';
      v[i].style.marginInline = 'auto';
    }
  }

  function tune() {
    var cap = 300;
    fit(cap);
    for (var i = 0; i < 4 && height() > MAX; i++) { cap = Math.max(150, cap - (height() - MAX)); fit(cap); }
  }

  function post() {
    tune();
    var h = height();
    if (h && Math.abs(h - last) > 1) { last = h; parent.postMessage({ wh: h, src: location.href }, '*'); }
  }

  addEventListener('load', function () { post(); setTimeout(post, 400); setTimeout(post, 1200); });
  addEventListener('resize', post);
  tune();
})();
