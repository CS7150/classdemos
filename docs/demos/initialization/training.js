// The live training demo shared by training-failure.html (tanh only) and
// activations.html, which adds an #actSeg switch between tanh, sigmoid and relu.

// ---------- data: two moons ----------
function mulberry32(a){ return function(){ a|=0;a=a+0x6D2B79F5|0; var t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
var rndData = mulberry32(7);
function randnSeeded(rng){ var u=0,v=0; while(u===0)u=rng(); while(v===0)v=rng(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
var DATA=[];
(function(){
  var n=60;
  for(var i=0;i<n;i++){
    var t=Math.PI*rndData();
    DATA.push({x:[Math.cos(t)+0.06*randnSeeded(rndData), Math.sin(t)+0.06*randnSeeded(rndData)], y:0});
    DATA.push({x:[1-Math.cos(t)+0.06*randnSeeded(rndData), 1-Math.sin(t)-0.5+0.06*randnSeeded(rndData)], y:1});
  }
})();

// ---------- activations ----------
// lr is the default learning rate; dfa is the derivative written in terms of the output a = f(h), which is all
// backprop needs; lo..hi is the range the activation histograms cover, and ylo..yhi
// the vertical range of the nonlinearity plot
var ACTS={
  tanh:   {f:Math.tanh, dfa:function(a){ return 1-a*a; }, lo:-1, hi:1, ylo:-1, yhi:1, lr:0.6},
  sigmoid:{f:function(h){ return 1/(1+Math.exp(-h)); }, dfa:function(a){ return a*(1-a); }, lo:0, hi:1, ylo:0, yhi:1, lr:0.6},
  // softsign h/(1+|h|): tanh's shape, but its tails flatten polynomially, not exponentially
  softsign:{f:function(h){ return h/(1+Math.abs(h)); }, dfa:function(a){ var q=1-Math.abs(a); return q*q; }, lo:-1, hi:1, ylo:-1, yhi:1, lr:0.6},
  // relu diverges at 0.6: its slope does not shrink as h grows, so big steps compound
  relu:   {f:function(h){ return h>0?h:0; }, dfa:function(a){ return a>0?1:0; }, lo:0, hi:3, ylo:0, yhi:5, lr:0.2}
};
var ACT='tanh';
function actF(h){ return ACTS[ACT].f(h); }
function actDf(h){ return ACTS[ACT].dfa(ACTS[ACT].f(h)); }
// saturated means the local slope is below 0.01: tanh or sigmoid pinned in a
// flat tail, or a relu that is off (dead)
function isSat(a){ return ACTS[ACT].dfa(a)<0.01; }
// activation mapped to [-1, 1] around the activation's resting value, for node colors
function centered(a){ return ACT==='sigmoid'?2*a-1:ACT==='relu'?Math.min(1,a):a; }
// for relu, zero on one input is normal ("off"); a unit is dead only if it is off on every input
var satWord=function(){ return ACT==='relu'?'off':'sat'; };

// ---------- network ----------
var randnG = function(){ return randnSeeded(Math.random); };
var WIDTH=10, depth=6, initStd;   // initStd is read from the slider
var net=null, lossHist=[], stepCount=0;

function buildSizes(d){ var s=[2]; for(var i=0;i<d;i++) s.push(WIDTH); s.push(1); return s; }

function initNet(){
  var sizes=buildSizes(depth);
  var W=[], b=[];
  for(var l=0;l<sizes.length-1;l++){
    var nin=sizes[l], nout=sizes[l+1];
    var std=initStd;   // every weight ~ N(0, initStd^2)
    var Wl=[]; for(var o=0;o<nout;o++){ var row=[]; for(var i=0;i<nin;i++) row.push(randnG()*std); Wl.push(row); }
    W.push(Wl); b.push(new Array(nout).fill(0));
  }
  net={sizes:sizes,W:W,b:b};
  lossHist=[]; stepCount=0;
}

function forward(x, net, keep, pres){   // pres, if given, collects each layer's pre-activations
  var a=x.slice();
  var acts = keep ? [a.slice()] : null;
  for(var l=0;l<net.W.length;l++){
    var Wl=net.W[l], bl=net.b[l], nout=Wl.length, nin=Wl[0].length;
    var h=new Array(nout);
    for(var o=0;o<nout;o++){ var s=bl[o]; var row=Wl[o]; for(var i=0;i<nin;i++) s+=row[i]*a[i]; h[o]=s; }
    if(pres) pres.push(h.slice());
    var isLast = l===net.W.length-1;
    var an = isLast ? h.map(function(v){return 1/(1+Math.exp(-v));}) : h.map(actF);
    a=an;
    if(keep) acts.push(a.slice());
  }
  return keep ? acts : a;
}

// one full-batch step; with apply=false it only measures activations and gradients
function trainStep(apply){
  var sizes=net.sizes, L=net.W.length;
  var gW=net.W.map(function(Wl){return Wl.map(function(row){return row.map(function(){return 0;});});});
  var gb=net.b.map(function(bl){return bl.map(function(){return 0;});});
  var totalLoss=0;
  var allActs=[], allGrads=[];
  DATA.forEach(function(pt){
    var acts=forward(pt.x, net, true);
    allActs.push(acts);
    var grads=[]; allGrads.push(grads);   // grads[l] = dL/da for hidden layer l
    var p=acts[L][0];
    var pc=Math.min(1-1e-7,Math.max(1e-7,p));
    totalLoss += -(pt.y*Math.log(pc)+(1-pt.y)*Math.log(1-pc));
    var delta=[p-pt.y]; // dL/dh for sigmoid+BCE
    for(var l=L-1;l>=0;l--){
      var Wl=net.W[l], nout=Wl.length, nin=Wl[0].length, aPrev=acts[l];
      for(var o=0;o<nout;o++){
        gb[l][o]+=delta[o];
        var row=gW[l][o];
        for(var i=0;i<nin;i++) row[i]+=delta[o]*aPrev[i];
      }
      if(l>0){
        var dPrev=new Array(nin).fill(0);
        for(var o2=0;o2<nout;o2++){ var row2=Wl[o2]; for(var i2=0;i2<nin;i2++) dPrev[i2]+=row2[i2]*delta[o2]; }
        grads[l]=dPrev.slice();
        var aHere=acts[l];
        for(var i3=0;i3<nin;i3++) dPrev[i3]*= ACTS[ACT].dfa(aHere[i3]); // phi'(h), from phi(h)
        delta=dPrev;
      }
    }
  });
  lastActs=allActs; lastGrads=allGrads;
  if(apply===false) return;
  var N=DATA.length;
  for(var l2=0;l2<L;l2++){
    for(var o3=0;o3<net.W[l2].length;o3++){
      net.b[l2][o3]-=lr*gb[l2][o3]/N;
      for(var i4=0;i4<net.W[l2][o3].length;i4++) net.W[l2][o3][i4]-=lr*gW[l2][o3][i4]/N;
    }
  }
  stepCount++;
  lossHist.push(totalLoss/N);
}

var lastActs=null, lastGrads=null;
function stepOnce(n){ for(var i=0;i<n;i++) trainStep(); }

// ---------- UI wiring ----------
// the slider carries log10(sigma); Xavier for a 10->10 layer is sqrt(2/(10+10)) = 10^-0.5
function showStd(){
  initStd=Math.pow(10,parseFloat(document.getElementById('scale').value));
  document.getElementById('scaleVal').textContent=initStd<0.1?initStd.toPrecision(2):initStd.toFixed(2);
}
function onScale(){
  showStd();
  Array.from(document.getElementById('presetSeg').children).forEach(function(x){x.classList.remove('on');});
  resetAll();
}
function onDepth(){
  depth=parseInt(document.getElementById('depth').value,10);
  document.getElementById('depthVal').textContent=depth;
  resetAll();
}
document.getElementById('presetSeg').addEventListener('click', function(e){
  var b=e.target.closest('button'); if(!b) return;
  var vals={small:-2, xavier:-0.5, he:Math.log10(Math.sqrt(0.2)), big:0.65};   // sigma = 0.01, 0.32, 0.45, 4.5
  document.getElementById('scale').value=vals[b.dataset.p];
  Array.from(document.getElementById('presetSeg').children).forEach(function(x){x.classList.toggle('on',x===b);});
  showStd();
  resetAll();
});

function resetAll(){
  setRunning(false);
  initNet();
  runTarget=STEP_LIMIT;
  if(!topChosen) topView='tanh';
  trainStep(false);
  redraw();
}

var running=false, timer=null;
var STEP_LIMIT=400;   // steps per press of Train
var runTarget=STEP_LIMIT;
var topView='tanh';   // top-right panel: 'tanh' curve or 'loss' curve
var topChosen=false;  // once the viewer picks one, Train and Reset stop switching it
function setRunning(r){
  running=r;
  document.getElementById('run').textContent = r?'Pause':'Train';
  clearInterval(timer);
  if(r){
    if(!topChosen) topView='loss';
    // a finished run continues for another STEP_LIMIT steps
    if(stepCount>=runTarget) runTarget=(Math.floor(stepCount/STEP_LIMIT)+1)*STEP_LIMIT;
    timer=setInterval(function(){
      stepOnce(Math.min(3,runTarget-stepCount));
      if(stepCount>=runTarget) setRunning(false);
      redraw();
    }, 60);
  }
}
function toggleRun(){ setRunning(!running); }

var view='net';
document.getElementById('viewSeg').addEventListener('click', function(e){
  var b=e.target.closest('button'); if(!b) return;
  view=b.dataset.v;
  Array.from(document.getElementById('viewSeg').children).forEach(function(x){x.classList.toggle('on',x===b);});
  redraw();
});

// ---------- drawing ----------
var NS='http://www.w3.org/2000/svg';
function el(tag,attrs){var e=document.createElementNS(NS,tag);for(var k in attrs)e.setAttribute(k,attrs[k]);return e;}
function txt(g,x,y,cls,str,attrs){ var t=el('text',Object.assign({x:x,y:y,class:cls},attrs||{})); t.textContent=str; g.appendChild(t); return t; }

var xMinD=-1.6,xMaxD=2.6,yMinD=-1.6,yMaxD=1.6;
var bx0=10, by0=10, bw=260, bh=260;
function bpx(x){ return bx0+(x-xMinD)/(xMaxD-xMinD)*bw; }
function bpy(y){ return by0+bh-(y-yMinD)/(yMaxD-yMinD)*bh; }

// probe input for the network view: hover the boundary plot to move it, click to
// pin it there; a pin survives Reset and Train, and lets go only when the pointer
// leaves the plot and comes back. Hovering the network diagram picks out one
// hidden unit (hover.l, hover.u) on the tanh plot.
var probe=DATA[0].x.slice(), pinned=false, leftSincePin=false, hover=null;
function netColX(l){ return PX0+14+l*(PW-28)/(net.sizes.length-1); }
function netNodeY(l,i){ var sp=Math.min(15,(PH-24)/(WIDTH-1)); return PY0+(PH-10)/2+(i-(net.sizes[l]-1)/2)*sp; }
function svgPoint(e){
  var svg=document.getElementById('stage'), pt=svg.createSVGPoint();
  pt.x=e.clientX; pt.y=e.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}
function inBoundary(q){ return q.x>=bx0&&q.x<=bx0+bw&&q.y>=by0&&q.y<=by0+bh; }
function setProbe(q){ probe=[xMinD+(q.x-bx0)/bw*(xMaxD-xMinD), yMinD+(by0+bh-q.y)/bh*(yMaxD-yMinD)]; }
function hoverAt(q){
  // null outside the network panel; {l:0} inside it; {l,u} on a hidden node
  if(q.x<PX0||q.x>PX0+PW||q.y<PY0-10||q.y>PY0+PH) return null;
  var best={l:0}, bd=8*8;
  for(var l=1;l<net.sizes.length-1;l++) for(var u=0;u<WIDTH;u++){
    var dx=q.x-netColX(l), dy=q.y-netNodeY(l,u);
    if(dx*dx+dy*dy<bd){ bd=dx*dx+dy*dy; best={l:l,u:u}; }
  }
  return best;
}
function sameHover(a,b){ return a===b||(a&&b&&a.l===b.l&&a.u===b.u); }
document.getElementById('stage').addEventListener('pointermove', function(e){
  if(view!=='net') return;
  var q=svgPoint(e), dirty=false;
  if(inBoundary(q)){
    if(pinned&&leftSincePin) pinned=false;
    if(!pinned){ setProbe(q); dirty=true; }
  } else leftSincePin=true;
  var h=hoverAt(q);
  if(!sameHover(h,hover)){ hover=h; dirty=true; }
  if(dirty&&!running) redraw();
});
// pointerdown, not click, so pinning works mid-training while the figure is rebuilt
document.getElementById('stage').addEventListener('pointerdown', function(e){
  if(view!=='net') return;
  var q=svgPoint(e);
  if(inBoundary(q)){ setProbe(q); pinned=true; leftSincePin=false; if(!running) redraw(); }
});
document.getElementById('stage').addEventListener('pointerleave', function(){
  leftSincePin=true;
  if(hover){ hover=null; if(!running) redraw(); }
});


function hiddenStats(){
  // per hidden layer: all activations over the dataset, std, saturated fraction
  var out=[];
  for(var l=1;l<net.sizes.length-1;l++){
    var vals=[];
    lastActs.forEach(function(acts){ for(var k=0;k<acts[l].length;k++) vals.push(acts[l][k]); });
    var m=0; vals.forEach(function(v){m+=v;}); m/=vals.length;
    var v2=0, sat=0; vals.forEach(function(v){ v2+=(v-m)*(v-m); if(isSat(v)) sat++; });
    var dead=0; for(var u=0;u<net.sizes[l];u++) if(lastActs.every(function(acts){ return isSat(acts[l][u]); })) dead++;
    out.push({vals:vals, std:Math.sqrt(v2/vals.length), sat:sat/vals.length, dead:dead/net.sizes[l]});
  }
  return out;
}

var PX0=300, PY0=150, PW=450, PH=170;   // bottom-right panel
function drawHistograms(g, stats){
  var n=stats.length, colW=PW/n, bins=20, histTop=PY0+4, histBot=PY0+PH-38;
  stats.forEach(function(st,c){
    var x0=PX0+c*colW, innerW=colW-8;
    var counts=new Array(bins).fill(0), satCounts=new Array(bins).fill(0);
    st.vals.forEach(function(v){
      var A=ACTS[ACT], bi=Math.min(bins-1,Math.max(0,Math.floor((v-A.lo)/(A.hi-A.lo)*bins)));
      counts[bi]++; if(isSat(v)) satCounts[bi]++;
    });
    var mx=Math.max.apply(null,counts);
    g.appendChild(el('line',{x1:x0+4,y1:histBot,x2:x0+4+innerW,y2:histBot,stroke:'var(--border-strong)','stroke-width':1}));
    g.appendChild(el('line',{x1:x0+4+innerW/2,y1:histBot,x2:x0+4+innerW/2,y2:histBot+3,stroke:'var(--border-strong)','stroke-width':1}));
    var bwid=innerW/bins;
    for(var b=0;b<bins;b++){
      if(!counts[b]) continue;
      var h=(histBot-histTop)*counts[b]/mx, satH=h*satCounts[b]/counts[b], bx=x0+4+b*bwid;
      if(satH>0.1) g.appendChild(el('rect',{x:bx+0.3,y:histBot-satH,width:bwid-0.6,height:satH,fill:'var(--red)',opacity:0.65}));
      if(h-satH>0.1) g.appendChild(el('rect',{x:bx+0.3,y:histBot-h,width:bwid-0.6,height:h-satH,fill:'var(--green)',opacity:0.65}));
    }
    txt(g,x0+4+innerW/2,histBot+14,'tn','L'+(c+1),{'text-anchor':'middle',style:'font-size:12px'});
    txt(g,x0+4+innerW/2,histBot+27,'tn','σ '+(st.std<0.01?st.std.toExponential(0):st.std.toFixed(2)),{'text-anchor':'middle',style:'font-size:11px'});
    txt(g,x0+4+innerW/2,histBot+38,'tn',Math.round(100*st.sat)+'% '+satWord(),{'text-anchor':'middle',style:'font-size:11px',fill:st.sat>0.3?'var(--red)':'var(--text-muted)'});
  });
  txt(g,PX0,PY0-6,'ts gl','activations, one histogram per layer over ['+String(ACTS[ACT].lo).replace('-','−')+', '+ACTS[ACT].hi+(ACT==='relu'?'+':'')+']');
}

// gradients span ~20 decades; bin log10|dL/da| over a useful range, and pile
// everything outside it into red overflow bars at either end
var GMIN=-6, GMAX=0.5;
function drawGradients(g){
  var n=net.sizes.length-2, colW=PW/n, bins=26, histTop=PY0+4, histBot=PY0+PH-38;
  for(var c=0;c<n;c++){
    var l=c+1, x0=PX0+c*colW, innerW=colW-8;
    var bwid=(innerW-4)/(bins+2), ax0=x0+4+bwid+2, ax1=ax0+bins*bwid;   // under | axis | over
    var counts=new Array(bins).fill(0), under=0, over=0, mags=[];
    lastGrads.forEach(function(gr){ gr[l].forEach(function(v){
      var m=Math.abs(v); mags.push(m);
      var lg=m>0?Math.log10(m):-Infinity;
      if(lg<GMIN) under++; else if(lg>=GMAX) over++;
      else counts[Math.min(bins-1,Math.floor((lg-GMIN)/(GMAX-GMIN)*bins))]++;
    }); });
    mags.sort(function(a,b){return a-b;});
    var med=mags[Math.floor(mags.length/2)], mx=Math.max(under,over,Math.max.apply(null,counts));
    var hOf=function(k){ return (histBot-histTop)*k/mx; };
    var gx=function(lg){ return ax0+(lg-GMIN)/(GMAX-GMIN)*(ax1-ax0); };
    g.appendChild(el('line',{x1:ax0,y1:histBot,x2:ax1,y2:histBot,stroke:'var(--border-strong)','stroke-width':1}));
    for(var t=-4;t<=0;t+=2)   // ticks at 1e-4, 1e-2, 1 (the last one taller)
      g.appendChild(el('line',{x1:gx(t),y1:histBot,x2:gx(t),y2:histBot+(t===0?5:3),stroke:'var(--border-strong)','stroke-width':1}));
    for(var b=0;b<bins;b++){
      if(!counts[b]) continue;
      g.appendChild(el('rect',{x:ax0+b*bwid+0.2,y:histBot-hOf(counts[b]),width:bwid-0.4,height:hOf(counts[b]),fill:'var(--violet)',opacity:0.7}));
    }
    if(under) g.appendChild(el('rect',{x:x0+4,y:histBot-hOf(under),width:bwid,height:hOf(under),fill:'var(--red)',opacity:0.7}));
    if(over) g.appendChild(el('rect',{x:ax1+2,y:histBot-hOf(over),width:bwid,height:hOf(over),fill:'var(--red)',opacity:0.7}));
    var out=(under+over)/mags.length;
    txt(g,x0+4+innerW/2,histBot+14,'tn','L'+l,{'text-anchor':'middle',style:'font-size:12px'});
    txt(g,x0+4+innerW/2,histBot+27,'tn','med '+(med>=0.01&&med<100?med.toFixed(2):med.toExponential(0)),{'text-anchor':'middle',style:'font-size:11px'});
    txt(g,x0+4+innerW/2,histBot+38,'tn',Math.round(100*out)+'% out',{'text-anchor':'middle',style:'font-size:11px',fill:out>0.3?'var(--red)':'var(--text-muted)'});
  }
  txt(g,PX0,PY0-6,'ts gl','gradient |∂L/∂a| per layer, log axis 10⁻⁶ … 1; red = outside');
}

// a two-button toggle for the top-right panel, right-aligned at (xr, y)
function drawTopToggle(g, xr, y){
  var opts=[['loss','loss'],['tanh',ACT]], bw=Math.max(40,7*ACT.length+10), bh=17, x=xr-bw*opts.length;
  opts.forEach(function(o,i){
    var on=topView===o[0], bg=el('g',{style:'cursor:pointer'});
    bg.appendChild(el('rect',{x:x+i*bw,y:y,width:bw,height:bh,rx:0,fill:on?'var(--surface-1)':'var(--surface)',stroke:'var(--border-strong)','stroke-width':0.8}));
    if(on) bg.appendChild(el('line',{x1:x+i*bw+1,y1:y+bh-1,x2:x+(i+1)*bw-1,y2:y+bh-1,stroke:'var(--blue)','stroke-width':1.6}));
    var t=txt(bg,x+i*bw+bw/2,y+12.5,'tn',o[1],{'text-anchor':'middle',style:'font-size:12px'+(on?';fill:var(--text-primary)':'')});
    // pointerdown, not click: during training the figure is rebuilt every frame, so the
    // button under pointerup is often a new element and a click would never fire
    bg.addEventListener('pointerdown',function(e){ e.preventDefault(); topView=o[0]; topChosen=true; redraw(); });
    g.appendChild(bg);
  });
}

// deterministic jitter in [-1,1), so points hold still as the probe moves
function jit(a,b,c){ var h=Math.imul(a*73856093^b*19349663^c*83492791^0x9e3779b9,0x85ebca6b); h^=h>>>13; h=Math.imul(h,0xc2b2ae35); h^=h>>>16; return (h>>>0)/2147483648-1; }
// isotropic gaussian jitter of std sig pixels, as an [dx, dy] pair (Box-Muller)
function gjit(a,b,c,sig){
  var r=sig*Math.sqrt(-2*Math.log(Math.max(1e-9,(jit(a,b,c)+1)/2))), t=Math.PI*(jit(c,a,b)+1);
  return [r*Math.cos(t), r*Math.sin(t)];
}
// hidden layers run from violet (L1) to green (deepest)
function layerColor(l,n){ return 'color-mix(in oklab, var(--green-ink) '+Math.round(100*(n>1?(l-1)/(n-1):0))+'%, var(--violet-ink))'; }

var HMAX=5;   // pre-activation axis of the nonlinearity plot; beyond it, points pile at the edge
// the activation curve (or, in gradients mode, its derivative) with every hidden unit on it
function drawTanh(g, x0, y0, w, h){
  var n=net.sizes.length-2, deriv=view==='grad', A=ACTS[ACT];
  var f=deriv ? actDf : actF, ylo=deriv?0:A.ylo, yhi=deriv?1:A.yhi;
  var tx=function(v){ return x0+(Math.max(-HMAX,Math.min(HMAX,v))+HMAX)/(2*HMAX)*w; };
  var ty=function(a){ return y0+6+(yhi-a)/(yhi-ylo)*(h-12); };
  // saturated regions, slope < 0.01, shaded
  var run=null, K=200;
  for(var k=0;k<=K;k++){
    var v=-HMAX+2*HMAX*k/K, flat=k<K&&actDf(v)<0.01;
    if(flat&&run===null) run=v;
    if(!flat&&run!==null){ g.appendChild(el('rect',{x:tx(run),y:y0,width:tx(v)-tx(run),height:h,fill:'var(--red)',opacity:0.08})); run=null; }
  }
  g.appendChild(el('line',{x1:x0,y1:ty(0),x2:x0+w,y2:ty(0),stroke:'var(--border)','stroke-width':0.8}));
  g.appendChild(el('line',{x1:tx(0),y1:y0,x2:tx(0),y2:y0+h,stroke:'var(--border)','stroke-width':0.8}));
  var yt=deriv?[0,1]:A.ylo<0?[A.ylo,0,A.yhi]:ACT==='sigmoid'?[0,0.5,1]:[0,A.yhi];
  yt.forEach(function(a){ txt(g,x0-4,ty(a)+4,'tn gl',String(a).replace('-','−'),{'text-anchor':'end',style:'font-size:12px'}); });
  [-4,-2,2,4].forEach(function(v){ txt(g,tx(v),ty(0)+13,'tn gl',String(v).replace('-','−'),{'text-anchor':'middle',style:'font-size:11px'}); });
  txt(g,x0+w-3,ty(0)-4,'tn gl','h',{'text-anchor':'end',style:'font-size:12px;font-style:italic'});
  var pts=[]; for(var k2=0;k2<=160;k2++){ var v2=-HMAX+2*HMAX*k2/160; pts.push(tx(v2)+','+ty(f(v2))); }
  g.appendChild(el('polyline',{points:pts.join(' '),fill:'none',stroke:'var(--text-muted)','stroke-width':1.4}));
  var name=ACT+(deriv?'′(h)':'(h)');
  if(view==='net'){
    // one digit per hidden unit on the probe input, labeled by its layer
    var pres=[]; forward(probe,net,false,pres);
    var pick=null;
    for(var l=1;l<=n;l++) for(var u=0;u<WIDTH;u++){
      var z=pres[l-1][u], j=gjit(l,u,1,3), px=tx(z)+j[0], py=ty(Math.min(yhi,f(z)))+4+j[1];
      if(hover&&hover.l===l&&hover.u===u){ pick=[px,py,l]; continue; }
      txt(g,px,py,'num',String(l),{'text-anchor':'middle',style:'font-size:11px;font-weight:600',
        fill:hover?'var(--text-muted)':layerColor(l,n),opacity:hover?0.3:1});
    }
    if(pick) txt(g,pick[0],pick[1]+1,'num gl',String(pick[2]),{'text-anchor':'middle',style:'font-size:15px;font-weight:700',fill:layerColor(pick[2],n)});
    txt(g,x0,y0-3,'ts gl',name+' on the probe point; digit = layer');
  } else {
    // every hidden unit on every training point, as a jittered cloud of dots
    DATA.forEach(function(pt,d){
      var pres=[]; forward(pt.x,net,false,pres);
      for(var l=1;l<=n;l++) for(var u=0;u<WIDTH;u++){
        var z=pres[l-1][u];
        var j=gjit(l,u,d,3);
        g.appendChild(el('circle',{cx:tx(z)+j[0],cy:ty(Math.min(yhi,f(z)))+j[1],r:1.1,fill:layerColor(l,n),opacity:0.45}));
      }
    });
    txt(g,x0,y0-3,'ts gl',name+': all units on all 120 points');
    txt(g,x0+6,y0+13,'tn gl','L1 violet → L'+n+' green',{style:'font-size:12px'});
  }
}

function drawNetwork(g){
  var sizes=net.sizes, acts=forward(probe,net,true), L=sizes.length;
  var colX=netColX, nodeY=netNodeY;
  // weights first, so nodes sit on top
  for(var l=0;l<net.W.length;l++){
    var Wl=net.W[l];
    for(var o=0;o<Wl.length;o++) for(var i=0;i<Wl[o].length;i++){
      var w=Wl[o][i], aw=Math.abs(w);
      var op=Math.min(0.45,aw*0.8);
      if(op<0.01) continue;
      g.appendChild(el('line',{x1:colX(l),y1:nodeY(l,i),x2:colX(l+1),y2:nodeY(l+1,o),
        stroke:w>0?'var(--orange)':'var(--blue)','stroke-width':Math.min(1.3,0.35+aw*0.5),opacity:op}));
    }
  }
  for(var l2=0;l2<L;l2++){
    for(var k=0;k<sizes[l2];k++){
      var a=acts[l2][k];
      var hidden=l2>0&&l2<L-1, sat=hidden&&isSat(a);
      if(l2===0) a=Math.max(-1,Math.min(1,a/2));            // inputs: rescaled coordinates
      else if(l2===L-1) a=2*a-1;                           // output: sigmoid mapped to [-1,1]
      else a=centered(a);
      var x=colX(l2), y=nodeY(l2,k);
      g.appendChild(el('circle',{cx:x,cy:y,r:5,fill:'var(--surface)',stroke:sat?'var(--red)':'var(--border-strong)','stroke-width':sat?2:1}));
      g.appendChild(el('circle',{cx:x,cy:y,r:4.2,fill:a>0?'var(--orange)':'var(--blue)',opacity:Math.min(1,Math.abs(a))}));
      if(hover&&hover.l===l2&&hover.u===k) g.appendChild(el('circle',{cx:x,cy:y,r:7.5,fill:'none',stroke:'var(--text-primary)','stroke-width':1.6}));
    }
  }
  txt(g,colX(0),PY0+PH-2,'tn','x',{'text-anchor':'middle',style:'font-size:12px'});
  for(var l3=1;l3<L-1;l3++) txt(g,colX(l3),PY0+PH-2,'tn','L'+l3,{'text-anchor':'middle',style:'font-size:12px'});
  txt(g,colX(L-1),PY0+PH-2,'tn','out',{'text-anchor':'middle',style:'font-size:12px'});
  txt(g,PX0,PY0-6,'ts gl','network on probe ◎ (hover boundary to move, click to pin)');
}

function redraw(){
  var g=document.getElementById('g');
  g.innerHTML='';
  // panel 1: decision boundary
  var grid=22;
  for(var gx=0;gx<grid;gx++){
    for(var gy=0;gy<grid;gy++){
      var xx=xMinD+(gx+0.5)/grid*(xMaxD-xMinD);
      var yy=yMinD+(gy+0.5)/grid*(yMaxD-yMinD);
      var p=forward([xx,yy],net,false)[0];
      var col = p>0.5 ? 'var(--orange)' : 'var(--blue)';
      var op = 0.12+0.35*Math.abs(p-0.5)*2;
      g.appendChild(el('rect',{x:bx0+gx*(bw/grid), y:by0+bh-(gy+1)*(bh/grid), width:bw/grid+0.5, height:bh/grid+0.5, fill:col, opacity:op}));
    }
  }
  DATA.forEach(function(pt){
    g.appendChild(el('circle',{cx:bpx(pt.x[0]),cy:bpy(pt.x[1]),r:3.2,fill:pt.y?'var(--orange-ink)':'var(--blue-ink)',stroke:'var(--surface)','stroke-width':1}));
  });
  g.appendChild(el('rect',{x:bx0,y:by0,width:bw,height:bh,fill:'none',stroke:'var(--border-strong)','stroke-width':1}));
  txt(g,bx0,by0-3,'ts gl','decision boundary');
  if(view==='net'){
    g.appendChild(el('circle',{cx:bpx(probe[0]),cy:bpy(probe[1]),r:7,fill:'none',stroke:'var(--text-primary)','stroke-width':1.6}));
    g.appendChild(el('circle',{cx:bpx(probe[0]),cy:bpy(probe[1]),r:1.6,fill:'var(--text-primary)'}));
  }

  // panel 2: the tanh curve with every hidden unit on it (shown on reset), or
  // the loss curve (shown once training starts), whose x axis widens as training continues
  var lx0=300, ly0=12, lw=450, lh=100;
  drawTopToggle(g, lx0+lw, ly0-21);
  if(topView==='tanh') drawTanh(g, lx0, ly0, lw, lh); else {
  var span=Math.max(STEP_LIMIT, runTarget, stepCount);
  g.appendChild(el('line',{x1:lx0,y1:ly0,x2:lx0,y2:ly0+lh,stroke:'var(--border-strong)','stroke-width':1}));
  g.appendChild(el('line',{x1:lx0,y1:ly0+lh,x2:lx0+lw,y2:ly0+lh,stroke:'var(--border-strong)','stroke-width':1}));
  var logMin=-2, logMax=0.15;
  function lly(v){ var l=Math.log10(Math.max(v,1e-6)); l=Math.max(logMin,Math.min(logMax,l)); return ly0+lh-(l-logMin)/(logMax-logMin)*lh; }
  [0.01,0.1,1].forEach(function(v){
    var y=lly(v);
    g.appendChild(el('line',{x1:lx0,y1:y,x2:lx0+lw,y2:y,stroke:'var(--border)','stroke-width':0.6,opacity:0.5}));
    txt(g,lx0-4,y+3,'tn gl',String(v),{'text-anchor':'end'});
  });
  for(var s=STEP_LIMIT;s<span;s+=STEP_LIMIT){
    var sx=lx0+s/span*lw;
    g.appendChild(el('line',{x1:sx,y1:ly0,x2:sx,y2:ly0+lh,stroke:'var(--border)','stroke-width':0.6,'stroke-dasharray':'3 3'}));
  }
  if(lossHist.length>1){
    var pts=lossHist.map(function(v,i){ return (lx0+(i+1)/span*lw)+','+lly(v); }).join(' ');
    g.appendChild(el('polyline',{points:pts,fill:'none',stroke:'var(--blue-ink)','stroke-width':1.8}));
  }
  txt(g,lx0,ly0-3,'ts gl','training loss (log scale), step '+stepCount+' / '+span);
  }

  // panel 3: hidden activations, as histograms or as the network
  var stats=hiddenStats();
  if(view==='net') drawNetwork(g); else if(view==='grad') drawGradients(g); else drawHistograms(g, stats);

  // readout
  var lastLoss = lossHist.length?lossHist[lossHist.length-1]:NaN;
  var deep = stats[stats.length-1], finalStd=deep.std, finalSat=deep.sat, finalDead=deep.dead;
  var verdict;
  if(finalStd<0.02) verdict='<span class="warn">vanishing &mdash; deep activations are nearly flat</span>';
  else if(ACT==='relu'&&finalDead>0.5) verdict='<span class="warn">dying &mdash; most units output 0 on every input</span>';
  else if(ACT!=='relu'&&finalSat>0.5) verdict='<span class="warn">saturating &mdash; units pinned in a flat tail</span>';
  else verdict='<span class="ok">healthy range</span>';
  document.getElementById('readout').innerHTML = 'step '+stepCount+' &middot; loss = <b class="num">'+(isFinite(lastLoss)?lastLoss.toFixed(4):'&mdash;')+'</b> &middot; deepest hidden layer: std = <b class="num">'+finalStd.toExponential(2)+'</b>, '+(ACT==='relu'?'off = <b class="num">'+Math.round(100*finalSat)+'%</b>, dead units = <b class="num">'+Math.round(100*finalDead)+'%</b>':'saturated = <b class="num">'+Math.round(100*finalSat)+'%</b>')+' &mdash; '+verdict;
}

var actSeg=document.getElementById('actSeg');
if(actSeg) actSeg.addEventListener('click',function(e){
  var b=e.target.closest('button'); if(!b) return;
  ACT=b.dataset.a;
  setLr(ACTS[ACT].lr);
  Array.from(actSeg.children).forEach(function(x){ x.classList.toggle('on',x===b); });
  resetAll();
});

// learning rate: a log slider; changing it mid-run just changes later steps
var lr;
function showLr(){ document.getElementById('lrVal').textContent=lr<0.1?lr.toPrecision(2):lr<10?lr.toFixed(2):lr.toFixed(1); }
function onLr(){ lr=Math.pow(10,parseFloat(document.getElementById('lr').value)); showLr(); }
function setLr(v){ lr=v; document.getElementById('lr').value=Math.log10(v); showLr(); }

setLr(ACTS[ACT].lr);
showStd();
resetAll();
