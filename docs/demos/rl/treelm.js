// A language model small enough to see all of it: four tokens, three steps,
// 64 possible completions.  The policy is a table of logits, one softmax per
// prefix (21 prefixes); a real LM computes the same per-prefix softmaxes with a
// shared network.  Shared by the RL explainers in this folder.
(function(){
var V=4,L=3,NN=21,NL=64,TOK=['a','b','c','d'];
var TCOL=['var(--blue)','var(--warm)','var(--green)','var(--pink)'];
var TINK=['var(--blue-ink)','var(--warm-ink)','var(--green-ink)','var(--pink-ink)'];
var OFF=[0,1,5];                       // first node id at each depth
function nodeOf(leaf,depth){return OFF[depth]+(depth===0?0:Math.floor(leaf/Math.pow(V,L-depth)));}
function tokOf(leaf,depth){return Math.floor(leaf/Math.pow(V,L-1-depth))%V;}
function leafStr(leaf){return TOK[tokOf(leaf,0)]+TOK[tokOf(leaf,1)]+TOK[tokOf(leaf,2)];}
function rng(seed){var a=seed>>>0;return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function gauss(r){var u=1-r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}

// softmax of every node: returns Float64Array(NN*V)
function probs(logits){var p=new Float64Array(NN*V);for(var n=0;n<NN;n++){var m=-1e300,s=0,k;
 for(k=0;k<V;k++)if(logits[n*V+k]>m)m=logits[n*V+k];
 for(k=0;k<V;k++){p[n*V+k]=Math.exp(logits[n*V+k]-m);s+=p[n*V+k];}for(k=0;k<V;k++)p[n*V+k]/=s;}return p;}
function leafProbs(p){var q=new Float64Array(NL);for(var y=0;y<NL;y++){var v=1;for(var d=0;d<L;d++)v*=p[nodeOf(y,d)*V+tokOf(y,d)];q[y]=v;}return q;}
function sample(p,r){var y=0;for(var d=0;d<L;d++){var n=OFF[d]+(d===0?0:y),u=r(),k=0,c=p[n*V];while(u>c&&k<V-1){k++;c+=p[n*V+k];}y=y*V+k;}return y;}
// g += coef * d/dlogits log pi(leaf)
function addGrad(g,p,leaf,coef){for(var d=0;d<L;d++){var n=nodeOf(leaf,d),t=tokOf(leaf,d);for(var k=0;k<V;k++)g[n*V+k]+=coef*((k===t?1:0)-p[n*V+k]);}}

// the "pretrained" reference model: a bigram preference plus noise, fixed
function refLogits(){var r=rng(7150),B=[],k,j;for(j=0;j<=V;j++){B.push([]);for(k=0;k<V;k++)B[j].push(1.1*gauss(r));}
 var lg=new Float64Array(NN*V);
 for(var n=0;n<NN;n++){var d=n<1?0:n<5?1:2,prev=d===0?V:(n-OFF[d])%V;for(k=0;k<V;k++)lg[n*V+k]=B[prev][k]+0.5*gauss(r);}
 return lg;}

// rewards on the 64 completions, all in [0,1]
var PRESETS={
 endsd:{name:'ends in d',f:function(y){return tokOf(y,2)===3?1:0;}},
 hasc:{name:'contains c',f:function(y){for(var d=0;d<L;d++)if(tokOf(y,d)===2)return 1;return 0;}},
 counta:{name:'more a’s is better',f:function(y){var c=0;for(var d=0;d<L;d++)if(tokOf(y,d)===0)c++;return c/3;}},
 norep:{name:'no repeated letter',f:function(y){var a=tokOf(y,0),b=tokOf(y,1),c=tokOf(y,2);return a!==b&&b!==c&&a!==c?1:0;}},
 needle:{name:'one rare completion',f:null}
};
function rewardTable(key,ref){var R=new Float64Array(NL);
 if(key==='needle'){var q=leafProbs(probs(ref)),best=0;   // a completion the reference model rarely writes, about 1 in 200
  var bestd=1e9;for(var y=0;y<NL;y++){var dd=Math.abs(Math.log(q[y])-Math.log(0.005));if(dd<bestd){bestd=dd;best=y;}}R[best]=1;return R;}
 for(var y2=0;y2<NL;y2++)R[y2]=PRESETS[key].f(y2);return R;}

// pi*(y) proportional to pi_ref(y) exp(R(y)/beta); beta=0 keeps only the best completions
function optimal(qref,R,beta){var s=new Float64Array(NL),z=0,mx=-1e300,y;for(y=0;y<NL;y++)if(R[y]>mx)mx=R[y];
 for(y=0;y<NL;y++){s[y]=beta>0?qref[y]*Math.exp((R[y]-mx)/beta):(R[y]>=mx-1e-12?qref[y]:0);z+=s[y];}
 for(y=0;y<NL;y++)s[y]/=z;return s;}
// per-node conditional distributions of a distribution over leaves
function condOf(q){var p=new Float64Array(NN*V);for(var n=0;n<NN;n++){var d=n<1?0:n<5?1:2,pre=n-OFF[d],span=Math.pow(V,L-d),cs=span/V,tot=0,k;
 for(k=0;k<V;k++){var s=0;for(var y=pre*span+k*cs;y<pre*span+(k+1)*cs;y++)s+=q[y];p[n*V+k]=s;tot+=s;}
 for(k=0;k<V;k++)p[n*V+k]=tot>1e-300?p[n*V+k]/tot:1/V;}return p;}
function expect(q,R){var s=0;for(var y=0;y<NL;y++)s+=q[y]*R[y];return s;}
function kl(q,r){var s=0;for(var y=0;y<NL;y++)if(q[y]>1e-300)s+=q[y]*Math.log(q[y]/r[y]);return Math.max(0,s);}
function entropy(q){var s=0;for(var y=0;y<NL;y++)if(q[y]>1e-300)s-=q[y]*Math.log(q[y]);return s;}
// exact gradient of -beta*KL(pi||pi_ref) w.r.t. the logits, added into g
function addKLGrad(g,p,q,qref,beta){if(!beta)return;for(var y=0;y<NL;y++)if(q[y]>1e-300)addGrad(g,p,y,-beta*q[y]*Math.log(q[y]/qref[y]));}

// ------------------------------------------------------------ drawing
function T(x,y,s,cls,anc,fill,extra){return '<text class="'+(cls||'tn')+'" x="'+(+x).toFixed(1)+'" y="'+(+y).toFixed(1)+'"'+(anc?' text-anchor="'+anc+'"':'')+(fill?' style="fill:'+fill+'"':'')+(extra||'')+'>'+s+'</text>';}
function R(x,y,w,h,fill,extra){return '<rect x="'+(+x).toFixed(1)+'" y="'+(+y).toFixed(1)+'" width="'+Math.max(0,w).toFixed(1)+'" height="'+Math.max(0,h).toFixed(1)+'" fill="'+fill+'"'+(extra||'')+'/>';}
function Ln(x1,y1,x2,y2,stroke,w,extra){return '<line x1="'+(+x1).toFixed(1)+'" y1="'+(+y1).toFixed(1)+'" x2="'+(+x2).toFixed(1)+'" y2="'+(+y2).toFixed(1)+'" stroke="'+stroke+'" stroke-width="'+(w||1)+'"'+(extra||'')+'/>';}

// The tree figure.  o: {p, pref, pstar (conditionals), q, qref, qstar (leaves), R,
//  samples:[{y,a}] (advantage-signed paths), leafExtra(y,x,w,top) optional}
var TW=1120,TX0=58,TX1=1112,CW=(TX1-TX0)/NL,LEV=[8,92,176],NH=50,LEAFY=262,PY=338,PH=100;
function sq(v){return Math.sqrt(Math.max(0,v));}
function drawTree(o){
 var s='',n,d,k;
 function nodeBox(n){var d=n<1?0:n<5?1:2,pre=n-OFF[d],span=NL/Math.pow(V,d),cx=TX0+(pre+0.5)*span*CW,bw=[30,21,13][d],bx=cx-bw*V/2;return {d:d,cx:cx,bx:bx,bw:bw,y:LEV[d]};}
 // sample paths, under everything
 if(o.samples&&o.samples.length){var agg={};o.samples.forEach(function(sm){for(var d=0;d<L;d++){var key=d+':'+Math.floor(sm.y/Math.pow(V,L-1-d));agg[key]=(agg[key]||0)+sm.a;}});
  var mx=1e-9;for(var key in agg)mx=Math.max(mx,Math.abs(agg[key]));
  for(var key2 in agg){var dd=+key2.split(':')[0],pre=+key2.split(':')[1],a=agg[key2];   var par=nodeBox(OFF[dd]+(dd===0?0:Math.floor(pre/V))),t=pre%V,x1=par.bx+(t+0.5)*par.bw,y1=par.y+NH,x2,y2;
   if(dd<L-1){var ch=nodeBox(OFF[dd+1]+pre);x2=ch.cx;y2=ch.y-4;}else{x2=TX0+(pre+0.5)*CW;y2=LEAFY-2;}
   var col=a>0?'var(--green)':a<0?'var(--red)':'var(--text-muted)';
   s+=Ln(x1,y1,x2,y2,col,(1.2+5*Math.abs(a)/mx).toFixed(1),' stroke-linecap="round" opacity="0.55"');}}
 // edges
 for(n=0;n<NN;n++){var b=nodeBox(n);for(k=0;k<V;k++){var x1=b.bx+(k+0.5)*b.bw,y1=b.y+NH,x2,y2;
  if(b.d<L-1){var c=nodeBox(OFF[b.d+1]+(n-OFF[b.d])*V+k);x2=c.cx;y2=c.y-4;}else{x2=TX0+((n-OFF[2])*V+k+0.5)*CW;y2=LEAFY-2;}
  s+=Ln(x1,y1,x2,y2,'var(--border)',1);}}
 // node bar charts
 for(n=0;n<NN;n++){var b2=nodeBox(n);s+=R(b2.bx-3,b2.y-4,b2.bw*V+6,NH+8,'var(--surface)',' stroke="var(--border-strong)" rx="4"');
  for(k=0;k<V;k++){var i=n*V+k,x=b2.bx+k*b2.bw+1,w=b2.bw-2,h=o.p[i]*NH;
   s+=R(x,b2.y+NH-h,w,h,TCOL[k],' opacity="0.85"');
   if(o.pstar){var hs=o.pstar[i]*NH;s+=R(x+0.5,b2.y+NH-hs,w-1,hs,'none',' stroke="var(--text-primary)" stroke-width="1.2" stroke-dasharray="3 2"');}
   if(o.pref){var hr=o.pref[i]*NH;s+=Ln(x-1,b2.y+NH-hr,x+w+1,b2.y+NH-hr,'var(--text-secondary)',2);}
   if(b2.d===0||b2.d===1)s+=T(x+w/2,b2.y+NH+(b2.d===0?14:13),TOK[k],'tn','middle',TINK[k],' style="font-size:'+(b2.d===0?13:11)+'px"');}}
 // leaves: labels, reward cells, probability bars (square-root scale)
 for(var y=0;y<NL;y++){var lx=TX0+y*CW;
  for(d=0;d<L;d++){var tk=tokOf(y,d);s+=T(lx+CW/2,LEAFY+11+d*12,TOK[tk],'tn','middle',TINK[tk],' style="font-size:12px;font-weight:600"');}
  var rv=o.R[y];s+='<g class="leaf" data-y="'+y+'">'+R(lx+1.5,LEAFY+40,CW-3,13,rv>0?'color-mix(in oklab, var(--green) '+Math.round(20+80*rv)+'%, var(--surface))':'var(--surface)',' stroke="var(--border-strong)" rx="2"')+'</g>';}
 s+=T(TX0-6,LEAFY+11,'1st','tn','end')+T(TX0-6,LEAFY+23,'2nd','tn','end')+T(TX0-6,LEAFY+35,'3rd','tn','end')+T(TX0-6,LEAFY+51,'reward','tn','end');
 [0,0.01,0.1,0.5,1].forEach(function(v){var yy=PY+PH-sq(v)*PH;s+=Ln(TX0,yy,TX1,yy,'var(--border)',v?0.6:1)+T(TX0-6,yy+4,v===0?'0':String(v),'tn','end');});
 s+=T(TX0-6,PY-12,'P(y)','tn','end');
 for(y=0;y<NL;y++){var lx2=TX0+y*CW,h2=sq(o.q[y])*PH;s+=R(lx2+2,PY+PH-h2,CW-4,h2,'var(--violet)',' opacity="0.8"');
  if(o.qstar){var h3=sq(o.qstar[y])*PH;s+=R(lx2+2.5,PY+PH-h3,CW-5,h3,'none',' stroke="var(--text-primary)" stroke-width="1.2" stroke-dasharray="3 2"');}
  if(o.qref){var h4=sq(o.qref[y])*PH;s+=Ln(lx2+1,PY+PH-h4,lx2+CW-1,PH+PY-h4,'var(--text-secondary)',2);}
  if(o.leafExtra)s+=o.leafExtra(y,lx2,CW,PY+PH);}
 return s;}
function legendHTML(){return '<span class="sw" style="background:var(--violet);opacity:.8"></span>the policy &pi;<sub>&theta;</sub> &nbsp; '+
 '<span class="sw" style="height:2px;background:var(--text-secondary);vertical-align:3px"></span> the pretrained model &pi;<sub>ref</sub> &nbsp; '+
 '<span class="sw" style="border:1.2px dashed var(--text-primary);background:none"></span> the KL-regularized optimum &pi;* &nbsp; '+
 '<span class="sw" style="background:var(--green);opacity:.6"></span><span class="sw" style="background:var(--red);opacity:.6"></span>last batch, pushed up or down';}

// a small line chart: series [{d:[...], c:color, w, dash, label}], hlines [{v,c,label}]
function chart(x0,y0,W,H,series,o){
 var s='',n=1,mx=o.ymax,mn=o.ymin||0;series.forEach(function(se){n=Math.max(n,se.d.length);});n=Math.max(n,o.minN||50);
 if(mx==null){mx=1e-6;series.forEach(function(se){se.d.forEach(function(v){if(v>mx)mx=v;});});(o.hlines||[]).forEach(function(h){if(h.v>mx)mx=h.v;});mx*=1.1;}
 s+=T(x0,y0-10,o.title,'ts','start');
 s+=R(x0,y0,W,H,'var(--surface)',' opacity="0.6" stroke="var(--border)"');
 [mn,(mn+mx)/2,mx].forEach(function(v){var yy=y0+H-(v-mn)/(mx-mn)*H;s+=T(x0-6,yy+4,o.fmt?o.fmt(v):v.toFixed(2),'tn','end');});
 (o.hlines||[]).forEach(function(h){var yy=y0+H-(Math.min(mx,h.v)-mn)/(mx-mn)*H;s+=Ln(x0,yy,x0+W,yy,h.c,1.4,' stroke-dasharray="5 4"');if(h.label)s+=T(x0+W-4,yy-5,h.label,'tn','end',h.c);});
 series.forEach(function(se){if(se.d.length<2)return;var step=Math.max(1,Math.floor(se.d.length/400)),pts=[];
  for(var i=0;i<se.d.length;i+=step)pts.push((x0+i/(n-1)*W).toFixed(1)+','+(y0+H-(Math.max(mn,Math.min(mx,se.d[i]))-mn)/(mx-mn)*H).toFixed(1));
  s+='<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+se.c+'" stroke-width="'+(se.w||2.2)+'"'+(se.dash?' stroke-dasharray="'+se.dash+'"':'')+' stroke-linejoin="round"/>';});
 s+=T(x0,y0+H+16,'0','tn','middle')+T(x0+W,y0+H+16,String(n-1),'tn','middle')+T(x0+W/2,y0+H+16,o.xlabel||'steps','tn','middle');
 return s;}

window.TreeLM={V:V,L:L,NN:NN,NL:NL,TOK:TOK,TCOL:TCOL,TINK:TINK,rng:rng,probs:probs,leafProbs:leafProbs,sample:sample,addGrad:addGrad,
 refLogits:refLogits,PRESETS:PRESETS,rewardTable:rewardTable,optimal:optimal,condOf:condOf,expect:expect,kl:kl,entropy:entropy,addKLGrad:addKLGrad,
 leafStr:leafStr,tokOf:tokOf,drawTree:drawTree,legendHTML:legendHTML,chart:chart,T:T,R:R,Ln:Ln,TW:TW,TX0:TX0,CW:CW,PY:PY,PH:PH};
})();
