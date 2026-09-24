// A five-layer, one-neuron-per-layer chain  h_l = w_l a_{l-1},  a_l = phi(h_l),
// L = a_5, drawn as a forward/backward computation graph (forward values above
// each wire, gradients below), with one small phi plot per layer, a log-scale
// strip of how hard each weight gets pushed, and the chain-rule product for one
// chosen dL/dw_l written out factor by factor.  Used by chain-rule.html.

var NS='http://www.w3.org/2000/svg';
function el(tag,attrs){var e=document.createElementNS(NS,tag);for(var k in attrs)e.setAttribute(k,attrs[k]);return e;}
function txt(g,x,y,cls,str,attrs){ var t=el('text',Object.assign({x:x,y:y,class:cls},attrs||{})); t.textContent=str; g.appendChild(t); return t; }

var ACTS={
  tanh:   {f:Math.tanh, df:function(h){ var t=Math.tanh(h); return 1-t*t; }, lo:-1.15, hi:1.15, tex:'\\tanh'}
};
var act='tanh';
var DEPTH=5, SMALL=0.1;    // factors below SMALL in magnitude are flagged red

// weights are sigma * z_l with z_l fixed standard normals, so the sigma slider
// rescales the same draw smoothly; Resample draws new z's
var z=[], sigma=1, x0=1, pick=1;
function randn(){ var u=0,v=0; while(!u)u=Math.random(); while(!v)v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
function resample(){ z=[]; for(var l=0;l<DEPTH;l++) z.push(randn()); compute(); }

var F=null;   // forward and backward values
function compute(){
  var A=ACTS[act], w=z.map(function(v){ return sigma*v; });
  var a=[x0], h=[null], d=[null];
  for(var l=1;l<=DEPTH;l++){ h[l]=w[l-1]*a[l-1]; a[l]=A.f(h[l]); d[l]=A.df(h[l]); }
  // backward: dL/da_5 = 1; dL/dh = phi' * dL/da; dL/dw = a_{l-1} * dL/dh; dL/da_{l-1} = w * dL/dh
  var ga=[], gh=[], gw=[];
  ga[DEPTH]=1;
  for(var k=DEPTH;k>=1;k--){ gh[k]=d[k]*ga[k]; gw[k]=a[k-1]*gh[k]; ga[k-1]=w[k-1]*gh[k]; }
  F={w:w,a:a,h:h,d:d,ga:ga,gh:gh,gw:gw};
  draw();
}

function fmt(v){
  var m=Math.abs(v), s;
  if(m===0) s='0';
  else if(m>=100) s=v.toFixed(0);
  else if(m>=0.1) s=v.toFixed(2);
  else if(m>=0.01) s=v.toFixed(3);
  else s=v.toExponential(1).replace('e-','e−');
  return s.replace(/^-/,'−');
}

var GY=112;            // graph row centre
var S=70, LW=186;
function mulX(l){ return S+(l-1)*LW+40; }
function actX(l){ return mulX(l)+93; }
var XIN=34, XOUT=1010;

function draw(){
  var g=document.getElementById('g'); g.innerHTML='';
  var A=ACTS[act];
  // ---- computation graph ----
  txt(g,8,14,'ts gl','forward values above each wire, gradients ∂L/∂(·) below; click a layer to expand its ∂L/∂w');
  function wire(x1,x2,val,grad,small){
    g.appendChild(el('line',{x1:x1,y1:GY,x2:x2,y2:GY,stroke:'var(--border-strong)','stroke-width':1.4}));
    g.appendChild(el('path',{d:'M'+(x2-6)+','+(GY-4)+' L'+x2+','+GY+' L'+(x2-6)+','+(GY+4),fill:'none',stroke:'var(--border-strong)','stroke-width':1.4}));
    txt(g,(x1+x2)/2,GY-7,'num gl',fmt(val),{'text-anchor':'middle',style:'font-size:12.5px',fill:'var(--green-ink)'});
    txt(g,(x1+x2)/2,GY+18,'num gl',fmt(grad),{'text-anchor':'middle',style:'font-size:12.5px',fill:small?'var(--red)':'var(--pink-ink)'});
  }
  function node(x,label,hot){
    g.appendChild(el('circle',{cx:x,cy:GY,r:15,fill:'var(--surface)',stroke:hot?'var(--red)':'var(--text-secondary)','stroke-width':hot?2.2:1.3}));
    txt(g,x,GY+5,'',label,{'text-anchor':'middle',style:'font-size:'+(label.length>2?11:15)+'px',fill:'var(--text-primary)'});
  }
  // input and output
  txt(g,XIN,GY+5,'',"x",{'text-anchor':'middle',style:'font-size:16px;font-style:italic',fill:'var(--text-primary)'});
  txt(g,XOUT+14,GY+5,'',"L",{'text-anchor':'middle',style:'font-size:16px;font-style:italic',fill:'var(--text-primary)'});
  for(var l=1;l<=DEPTH;l++){
    var xm=mulX(l), xa=actX(l), from=l===1?XIN+10:actX(l-1)+15;
    // column highlight for the picked layer
    if(l===pick) g.appendChild(el('rect',{x:xm-34,y:22,width:actX(l)-xm+68,height:430,rx:8,fill:'var(--blue)',opacity:0.06}));
    wire(from,xm-15,F.a[l-1],F.ga[l-1],Math.abs(F.ga[l-1])<SMALL*SMALL);
    wire(xm+15,xa-15,F.h[l],F.gh[l],Math.abs(F.gh[l])<SMALL*SMALL);
    // the weight enters the multiply node from above
    g.appendChild(el('line',{x1:xm,y1:GY-54,x2:xm,y2:GY-16,stroke:'var(--border-strong)','stroke-width':1.2}));
    txt(g,xm,GY-72,'num gl','w'+'₁₂₃₄₅'[l-1]+' = '+fmt(F.w[l-1]),{'text-anchor':'middle',style:'font-size:13px',fill:'var(--green-ink)'});
    var gwSmall=Math.abs(F.gw[l])<SMALL*SMALL;
    txt(g,xm,GY-57,'num gl','∂L/∂w = '+fmt(F.gw[l]),{'text-anchor':'middle',style:'font-size:12.5px;font-weight:600',fill:gwSmall?'var(--red)':'var(--pink-ink)'});
    node(xm,'×',false);
    node(xa,'tanh',F.d[l]<SMALL);
  }
  wire(actX(DEPTH)+15,XOUT,F.a[DEPTH],1,false);

  // ---- one small phi plot per layer ----
  var PY=190, PH=112, PW=150;
  for(var l=1;l<=DEPTH;l++){
    var cx=(mulX(l)+actX(l))/2, px0=cx-PW/2, HM=4;
    var tx=function(v){ return px0+(Math.max(-HM,Math.min(HM,v))+HM)/(2*HM)*PW; };
    var ty=function(a){ return PY+PH-(a-A.lo)/(A.hi-A.lo)*PH; };
    g.appendChild(el('line',{x1:actX(l),y1:GY+24,x2:actX(l),y2:PY-4,stroke:'var(--border)','stroke-width':1,'stroke-dasharray':'2 3'}));
    g.appendChild(el('rect',{x:px0,y:PY,width:PW,height:PH,fill:'var(--surface)',stroke:'var(--border)','stroke-width':1}));
    g.appendChild(el('line',{x1:px0,y1:ty(0),x2:px0+PW,y2:ty(0),stroke:'var(--border)','stroke-width':0.8}));
    g.appendChild(el('line',{x1:tx(0),y1:PY,x2:tx(0),y2:PY+PH,stroke:'var(--border)','stroke-width':0.8}));
    var pts=[]; for(var k=0;k<=80;k++){ var v=-HM+2*HM*k/80; pts.push(tx(v)+','+ty(A.f(v))); }
    g.appendChild(el('polyline',{points:pts.join(' '),fill:'none',stroke:'var(--text-muted)','stroke-width':1.4}));
    var hh=F.h[l], off=Math.abs(hh)>HM, hot=F.d[l]<SMALL;
    // tangent line: its slope is the local derivative
    var hc=Math.max(-HM,Math.min(HM,hh)), ac=A.f(hc), sl=A.df(hc), dh=1.1;
    g.appendChild(el('line',{x1:tx(hc-dh),y1:ty(ac-sl*dh),x2:tx(hc+dh),y2:ty(ac+sl*dh),stroke:hot?'var(--red)':'var(--blue)','stroke-width':2,opacity:0.8}));
    g.appendChild(el('circle',{cx:tx(hc),cy:ty(ac),r:4.5,fill:hot?'var(--red)':'var(--blue-ink)',stroke:'var(--surface)','stroke-width':1.2}));
    if(off) txt(g,hh>0?px0+PW-4:px0+4,ty(ac)+(ac>0.5*(A.hi+A.lo)?16:-8),'tn gl','h = '+fmt(hh)+(hh>0?' →':''),{'text-anchor':hh>0?'end':'start',style:'font-size:11px'});
    txt(g,cx,PY+PH+15,'num',"slope tanh′(h"+'₁₂₃₄₅'[l-1]+") = "+fmt(F.d[l]),{'text-anchor':'middle',style:'font-size:12px',fill:hot?'var(--red)':'var(--text-secondary)'});
    if(hot) txt(g,cx,PY+14,'tn gl','saturated: flat',{'text-anchor':'middle',style:'font-size:11.5px',fill:'var(--red)'});
  }

  // ---- log-scale strip: |dL/dw_l| per layer, and |dL/dh_l| as a line ----
  var SY=360, SH=80, LMIN=-12, LMAX=1;
  var sy=function(v){ var lg=v>0?Math.log10(v):LMIN; lg=Math.max(LMIN,Math.min(LMAX,lg)); return SY+SH-(lg-LMIN)/(LMAX-LMIN)*SH; };
  txt(g,8,SY-10,'ts gl','how hard each weight gets pushed: |∂L/∂w| (bars) and |∂L/∂h| (line), log scale');
  [[1,'1'],[1e-4,'10⁻⁴'],[1e-8,'10⁻⁸'],[1e-12,'10⁻¹²']].forEach(function(t){
    g.appendChild(el('line',{x1:S,y1:sy(t[0]),x2:XOUT,y2:sy(t[0]),stroke:'var(--border)','stroke-width':0.7,'stroke-dasharray':t[0]===1?'':'3 3'}));
    txt(g,S-6,sy(t[0])+4,'tn gl',t[1],{'text-anchor':'end',style:'font-size:11.5px'});
  });
  var lpts=[];
  for(var l=1;l<=DEPTH;l++){
    var v=Math.abs(F.gw[l]), xm=mulX(l);
    g.appendChild(el('rect',{x:xm-16,y:sy(v),width:32,height:SY+SH-sy(v),fill:v<SMALL*SMALL?'var(--red)':'var(--pink)',opacity:0.7}));
    lpts.push(((mulX(l)+actX(l))/2)+','+sy(Math.abs(F.gh[l])));
  }
  g.appendChild(el('polyline',{points:lpts.join(' '),fill:'none',stroke:'var(--pink-ink)','stroke-width':1.6}));
  lpts.forEach(function(p){ var q=p.split(','); g.appendChild(el('circle',{cx:q[0],cy:q[1],r:2.6,fill:'var(--pink-ink)'})); });

  // clickable columns
  for(var l=1;l<=DEPTH;l++){
    var r=el('rect',{x:mulX(l)-34,y:22,width:actX(l)-mulX(l)+68,height:430,fill:'transparent',style:'cursor:pointer'});
    (function(k){ r.addEventListener('pointerdown',function(){ pick=k; draw(); }); })(l);
    g.appendChild(r);
  }
  drawFormula();
}

// a number as TeX: plain decimals, or m \times 10^{e} for tiny ones
function tnum(v){
  var m=Math.abs(v);
  if(m===0||m>=0.01) return fmt(v).replace(/−/g,'-');
  var e=Math.floor(Math.log10(m)), mant=v/Math.pow(10,e);
  return mant.toFixed(1)+'\\times10^{'+e+'}';
}

// the chain-rule product for dL/dw_pick, factor by factor
function drawFormula(){
  var k=pick, A=ACTS[act], P=A.tex, parts=[], sub=function(i){ return '_{'+i+'}'; };
  function fac(tex,val){
    var s='\\underbrace{'+tex+'}_{'+tnum(val)+'}';
    return Math.abs(val)<SMALL?'\\textcolor{#E24B4A}{'+s+'}':s;
  }
  parts.push(fac('a'+sub(k-1),F.a[k-1]));
  parts.push(fac(P+"'(h"+sub(k)+')',F.d[k]));
  for(var j=k+1;j<=DEPTH;j++){ parts.push(fac('w'+sub(j),F.w[j-1])); parts.push(fac(P+"'(h"+sub(j)+')',F.d[j])); }
  var tex='\\frac{\\partial L}{\\partial w'+sub(k)+'}='+parts.join('\\cdot ')+'='+tnum(F.gw[k]);
  katex.render(tex, document.getElementById('formula'), {throwOnError:false, displayMode:true});
  var small=[];
  if(Math.abs(F.a[k-1])<SMALL) small.push('its input a'+(k-1)+' is near zero, so the product rule at the × node hands w'+k+' almost nothing');
  var sat=[]; for(var j=k;j<=DEPTH;j++) if(F.d[j]<SMALL) sat.push(j);
  if(sat.length) small.push('the slope is nearly flat'+' at layer'+(sat.length>1?'s ':' ')+sat.join(', '));
  var tiny=[]; for(var j2=k+1;j2<=DEPTH;j2++) if(Math.abs(F.w[j2-1])<SMALL) tiny.push(j2);
  if(tiny.length) small.push('the weight'+(tiny.length>1?'s':'')+' w'+tiny.join(', w')+' above '+(tiny.length>1?'are':'is')+' tiny');
  document.getElementById('readout').innerHTML='<b>∂L/∂w'+k+' = '+fmt(F.gw[k])+'</b> &mdash; '+
    (small.length?'<span style="color:var(--red)">'+(F.gw[k]===0?'exactly zero':'small')+' because '+small.join('; and ')+'.</span>'
      :Math.abs(F.gw[k])<SMALL*SMALL?'<span style="color:var(--red)">no single factor is below '+SMALL+', but '+(2*(DEPTH-k)+2)+' factors, each less than 1, compound.</span>'
      :'no factor is below '+SMALL+', so the gradient survives.');
}

// ---------- controls ----------
function showSigma(){ document.getElementById('sigVal').textContent=sigma<0.1?sigma.toPrecision(2):sigma.toFixed(2); }
function onSigma(){
  sigma=Math.pow(10,parseFloat(document.getElementById('sig').value)); showSigma();
  Array.from(document.getElementById('presetSeg').children).forEach(function(b){ b.classList.remove('on'); });
  compute();
}
function onX(){ x0=parseFloat(document.getElementById('xin').value); document.getElementById('xVal').textContent=fmt(x0); compute(); }
document.getElementById('presetSeg').addEventListener('click',function(e){
  var b=e.target.closest('button'); if(!b) return;
  document.getElementById('sig').value=Math.log10(parseFloat(b.dataset.s));
  sigma=parseFloat(b.dataset.s); showSigma();
  Array.from(this.children).forEach(function(x){ x.classList.toggle('on',x===b); });
  compute();
});
showSigma(); resample();
