var W=[0.906,0.423],U=[-0.423,0.906],G=0.15;
var Q=[[0.15,0.9],[0.15,-0.9],[0.20,0.4],[0.35,-0.5],[0.25,1.15],[0.45,0.15],[0.18,-0.35],[0.30,0.65]];
var XS=[],P=[];
for(var z=0;z<Q.length;z++){var qa=Q[z][0]*W[0]+Q[z][1]*U[0],qb=Q[z][0]*W[1]+Q[z][1]*U[1];XS.push([qa,qb]);P.push([qa,qb,1]);P.push([-qa,-qb,-1]);}
var LX=261,LY=196,RX=779,RY=196,LR=[12,30,510,358],RR=[530,30,1028,358];
var ls=88,rs=84,w,k,idx,hit,hs,phase,msg,TR,MET,timer=null;
function L(a,b){return [LX+ls*a,LY-ls*b];}
function Rp(a,b){return [RX+rs*a,RY-rs*b];}
function clip(p,q,r,inf){var dx=q[0]-p[0],dy=q[1]-p[1],t0=inf?-1e7:0,t1=inf?1e7:1,ok=true;
 function c(pp,qq){if(pp===0){if(qq<0)ok=false;return;}var rr=qq/pp;if(pp<0){if(rr>t1)ok=false;else if(rr>t0)t0=rr;}else{if(rr<t0)ok=false;else if(rr<t1)t1=rr;}}
 c(-dx,p[0]-r[0]);c(dx,r[2]-p[0]);c(-dy,p[1]-r[1]);c(dy,r[3]-p[1]);
 if(!ok)return null;return {a:[p[0]+t0*dx,p[1]+t0*dy],b:[p[0]+t1*dx,p[1]+t1*dy],full:t1>=0.999};}
function ln(p,q,r,st,inf,mk){var c=clip(p,q,r,inf);if(!c)return '';
 return '<line x1="'+c.a[0].toFixed(1)+'" y1="'+c.a[1].toFixed(1)+'" x2="'+c.b[0].toFixed(1)+'" y2="'+c.b[1].toFixed(1)+'" '+st+(mk&&c.full?' marker-end="url(#arrow)"':'')+'/>';}
function hp(poly,nx,ny,c){var o=[];for(var i=0;i<poly.length;i++){var A=poly[i],B=poly[(i+1)%poly.length],da=nx*A[0]+ny*A[1]-c,db=nx*B[0]+ny*B[1]-c;
 if(da>=0)o.push(A);if((da>=0)!==(db>=0)){var t=da/(da-db);o.push([A[0]+t*(B[0]-A[0]),A[1]+t*(B[1]-A[1])]);}}return o;}
function ge(poly,vx,vy,val){return hp(poly,vx,-vy,val*ls+vx*LX-vy*LY);}
function poly(pts,f,op){if(pts.length<3)return '';var s='';for(var i=0;i<pts.length;i++)s+=pts[i][0].toFixed(1)+','+pts[i][1].toFixed(1)+' ';
 return '<polygon points="'+s+'" fill="'+f+'" opacity="'+op+'"/>';}
function rect(r){return [[r[0],r[1]],[r[2],r[1]],[r[2],r[3]],[r[0],r[3]]];}
function bad(p){return p[2]*(w[0]*p[0]+w[1]*p[1])<=0;}
function corr(){var h=P[hit];return [h[2]*h[0],h[2]*h[1]];}
function feasAll(f){var c=Math.cos(f),s=Math.sin(f);for(var j=0;j<XS.length;j++)if(c*XS[j][0]+s*XS[j][1]<=0)return false;return true;}
function feasMet(f){var c=Math.cos(f),s=Math.sin(f);for(var j=0;j<MET.length;j++)if(c*MET[j][0]+s*MET[j][1]<=0)return false;return true;}
var p0=Math.atan2(W[1],W[0]),A1=p0,A2=p0,ST=0.0015;
while(feasAll(A1-ST))A1-=ST;while(feasAll(A2+ST))A2+=ST;
function metArc(){if(!MET.length)return null;var a=p0,b=p0,n=0;
 while(feasMet(a-ST)&&n++<2090)a-=ST;n=0;while(feasMet(b+ST)&&n++<2090)b+=ST;return [a,b];}
function wedge(a1,a2,r,f,op){var big=(a2-a1)>Math.PI?1:0;
 return '<path d="M '+RX+' '+RY+' L '+(RX+r*Math.cos(a1)).toFixed(1)+' '+(RY-r*Math.sin(a1)).toFixed(1)+' A '+r+' '+r+' 0 '+big+' 0 '+(RX+r*Math.cos(a2)).toFixed(1)+' '+(RY-r*Math.sin(a2)).toFixed(1)+' Z" fill="'+f+'" opacity="'+op+'"/>';}
function need(){var m=Math.hypot(w[0],w[1]);if(hs==='bad'&&hit!=null){var c=corr();m=Math.max(m,Math.hypot(w[0]+c[0],w[1]+c[1]));}return Math.max(m*1.18,1.35);}
function anim(){var tl=Math.max(42,Math.min(94,126/need())),tr=Math.max(12,Math.min(90,122/need()));
 if(Math.abs(tl-ls)>0.4||Math.abs(tr-rs)>0.4){ls+=(tl-ls)*0.22;rs+=(tr-rs)*0.22;render();requestAnimationFrame(anim);}else{ls=tl;rs=tr;render();}}
function render(){
 var s='',n=Math.hypot(w[0],w[1]);
 s+='<line x1="520" y1="26" x2="520" y2="360" stroke="var(--border-strong)" stroke-width="0.5"/>';
 s+='<text class="ts" x="261" y="18" text-anchor="middle" fill="var(--text-secondary)">data space</text>';
 s+='<text class="ts" x="779" y="18" text-anchor="middle" fill="var(--text-secondary)">weight space</text>';
 if(n>0)s+=poly(ge(rect(LR),w[0],w[1],0),'#378ADD',0.10);
 s+=poly(ge(ge(rect(LR),W[0],W[1],-G),-W[0],-W[1],-G),'#1D9E75',0.16);
 s+=ln(L(0,0),L(U[0],U[1]),LR,'stroke="#0F6E56" stroke-width="1" stroke-dasharray="5 4"',1);
 var nb=0;
 for(var i=0;i<P.length;i++){var p=P[i],c=L(p[0],p[1]);
  if(bad(p)){nb++;s+='<circle cx="'+c[0].toFixed(1)+'" cy="'+c[1].toFixed(1)+'" r="8" fill="none" stroke="#E24B4A" stroke-width="1" opacity="0.5"/>';}
  s+='<circle cx="'+c[0].toFixed(1)+'" cy="'+c[1].toFixed(1)+'" r="4.5" fill="'+(p[2]>0?'#378ADD':'var(--surface-2)')+'" stroke="'+(p[2]>0?'#185FA5':'#5F5E5A')+'" stroke-width="1"/>';}
 if(n>0)s+=ln(L(-w[1]/n,w[0]/n),L(w[1]/n,-w[0]/n),LR,'stroke="#BA7517" stroke-width="2"',1);
 if(n>0)s+=ln(L(0,0),L(w[0],w[1]),LR,'stroke="#BA7517" stroke-width="2.5"',0,1);
 if(hs==='bad'&&hit!=null){var h=P[hit],cr=corr(),pv=[w[0]+cr[0],w[1]+cr[1]],ph=L(h[0],h[1]);
  s+='<circle cx="'+ph[0].toFixed(1)+'" cy="'+ph[1].toFixed(1)+'" r="10.5" fill="none" stroke="#E24B4A" stroke-width="2.5"/>';
  if(h[2]<0)s+=ln(L(h[0],h[1]),L(cr[0],cr[1]),LR,'stroke="#D85A30" stroke-width="0.5" stroke-dasharray="3 3"',0);
  s+=ln(L(0,0),L(cr[0],cr[1]),LR,'stroke="#D85A30" stroke-width="2"',0,1);
  s+=ln(L(w[0],w[1]),L(pv[0],pv[1]),LR,'stroke="#D85A30" stroke-width="0.75" stroke-dasharray="5 4"',0);
  s+=ln(L(cr[0],cr[1]),L(pv[0],pv[1]),LR,'stroke="#BA7517" stroke-width="0.75" stroke-dasharray="5 4"',0);
  s+=ln(L(0,0),L(pv[0],pv[1]),LR,'stroke="#7F77DD" stroke-width="2" stroke-dasharray="7 5"',0,1);
  var lp=L(cr[0]*0.55,cr[1]*0.55);
  s+='<text class="ts" x="'+(lp[0]+10).toFixed(1)+'" y="'+(lp[1]-6).toFixed(1)+'" fill="#993C1D">y&#183;x</text>';}
 var ma=metArc();
 if(ma)s+=wedge(ma[0],ma[1],150,'#7F77DD',0.16);
 s+=wedge(A1,A2,150,'#1D9E75',0.30);
 for(var j=0;j<XS.length;j++){var x=XS[j],nn=Math.hypot(x[0],x[1]),pp=[-x[1]/nn,x[0]/nn],hi=(hs==='bad'&&hit!=null&&Math.abs(Math.abs(P[hit][0])-Math.abs(x[0]))<1e-9);
  s+=ln(Rp(pp[0],pp[1]),Rp(-pp[0],-pp[1]),RR,'stroke="'+(hi?'#E24B4A':'var(--border-strong)')+'" stroke-width="'+(hi?2:0.75)+'"',1);}
 var mid=(A1+A2)/2;
 s+='<text class="ts" x="'+(RX+72*Math.cos(mid)).toFixed(1)+'" y="'+(RY-72*Math.sin(mid)).toFixed(1)+'" text-anchor="middle" fill="#0F6E56">solves all 16</text>';
 if(ma)s+='<text class="ts" x="'+(RX+126*Math.cos(ma[1]-0.12)).toFixed(1)+'" y="'+(RY-126*Math.sin(ma[1]-0.12)).toFixed(1)+'" text-anchor="middle" fill="#5A4FBF">consistent so far</text>';
 if(TR.length>1){var pts='';for(var t2=0;t2<TR.length;t2++){var tp=Rp(TR[t2][0],TR[t2][1]);pts+=tp[0].toFixed(1)+','+tp[1].toFixed(1)+' ';}
  s+='<polyline points="'+pts+'" fill="none" stroke="#BA7517" stroke-width="1" stroke-dasharray="3 3" opacity="0.5"/>';}
 if(hs==='bad'&&hit!=null){var cr2=corr(),pv2=[w[0]+cr2[0],w[1]+cr2[1]];
  s+=ln(Rp(0,0),Rp(cr2[0],cr2[1]),RR,'stroke="#D85A30" stroke-width="2"',0,1);
  s+=ln(Rp(w[0],w[1]),Rp(pv2[0],pv2[1]),RR,'stroke="#D85A30" stroke-width="0.75" stroke-dasharray="5 4"',0);
  s+=ln(Rp(cr2[0],cr2[1]),Rp(pv2[0],pv2[1]),RR,'stroke="#BA7517" stroke-width="0.75" stroke-dasharray="5 4"',0);
  s+=ln(Rp(0,0),Rp(pv2[0],pv2[1]),RR,'stroke="#7F77DD" stroke-width="2" stroke-dasharray="7 5"',0,1);}
 if(n>0)s+=ln(Rp(0,0),Rp(w[0],w[1]),RR,'stroke="#BA7517" stroke-width="2.5"',0,1);
 s+='<circle cx="'+RX+'" cy="'+RY+'" r="2.5" fill="var(--text-primary)"/>';
 document.getElementById('stage').innerHTML=s;
 var mw=ma?(((ma[1]-ma[0])/2)*180/Math.PI).toFixed(0)+'°':'unbounded';
 document.getElementById('readout').innerHTML=msg+'<br>updates <b>k = '+k+'</b> · misclassified <b>'+nb+' of 16</b> · |w| = <b>'+n.toFixed(2)+'</b> · consistent region <b>'+mw+'</b> narrowing to '+(((A2-A1)/2)*180/Math.PI).toFixed(0)+'°';
}
function act(){var btn=document.getElementById('go');
 if(phase==='find'){
  for(var j=0;j<P.length;j++){var i=(idx+j)%P.length,p=P[i];
   if(bad(p)){hit=i;hs='bad';phase='apply';btn.textContent='Apply update';
    msg='On the left, the parallelogram: w plus the point itself. On the right, the red line is the constraint being violated.';
    anim();return true;}}
  hit=null;hs=null;msg='w is inside the green wedge. Nothing moves again.';anim();return false;}
 var c=corr();MET.push([c[0],c[1]]);w=[w[0]+c[0],w[1]+c[1]];TR.push([w[0],w[1]]);k++;idx=hit+1;hs='fixed';phase='find';
 btn.textContent='Find a mistake';msg='Committed. The violet wedge just narrowed by that constraint.';
 anim();return true;}
function reset(){w=[-1.214,-0.236];TR=[[w[0],w[1]]];MET=[];k=0;idx=0;hit=null;hs=null;phase='find';ls=88;rs=84;
 msg='w starts outside the wedge: 4 of 16 correct.';
 document.getElementById('go').textContent='Find a mistake';anim();}
reset();
