var OX=150,OY=170,g=0.30,t=3;
function KB(){return Math.floor(1/(g*g));}
function sc(){return 300*g;}
function tip(k){var s=sc(),a=k*g,b=Math.sqrt(Math.max(0,k-k*k*g*g));
 return [OX+s*a,OY-s*b*(k%2?1:-1)];}
function render(){
 var s='',S=sc(),kb=KB(),th=Math.acos(Math.min(1,Math.sqrt(t)*g)),wd=Math.asin(g),Rr=300;
 s+='<path d="M '+OX+' '+OY+' L '+(OX+Rr*Math.cos(th)).toFixed(1)+' '+(OY-Rr*Math.sin(th)).toFixed(1)+' A '+Rr+' '+Rr+' 0 0 0 '+(OX+Rr*Math.cos(th)).toFixed(1)+' '+(OY+Rr*Math.sin(th)).toFixed(1)+' Z" fill="#378ADD" opacity="0.15"/>';
 s+='<path d="M '+OX+' '+OY+' L '+(OX+Rr*Math.cos(wd)).toFixed(1)+' '+(OY-Rr*Math.sin(wd)).toFixed(1)+' A '+Rr+' '+Rr+' 0 0 0 '+(OX+Rr*Math.cos(wd)).toFixed(1)+' '+(OY+Rr*Math.sin(wd)).toFixed(1)+' Z" fill="#1D9E75" opacity="0.32"/>';
 s+='<line x1="'+(OX-30)+'" y1="'+OY+'" x2="640" y2="'+OY+'" stroke="var(--border-strong)" stroke-width="0.5"/>';
 s+='<text class="ts" x="600" y="'+(OY+20)+'" text-anchor="end" fill="#0F6E56">direction of w*</text>';
 for(var k=1;k<=kb;k++){var x=OX+S*k*g;if(x<650)s+='<line x1="'+x.toFixed(1)+'" y1="'+(OY-4)+'" x2="'+x.toFixed(1)+'" y2="'+(OY+4)+'" stroke="var(--text-muted)" stroke-width="0.5"/>';}
 var pts='';for(var k2=0;k2<=t;k2++){var p=k2===0?[OX,OY]:tip(k2);pts+=p[0].toFixed(1)+','+p[1].toFixed(1)+' ';}
 s+='<polyline points="'+pts+'" fill="none" stroke="#BA7517" stroke-width="2"/>';
 for(var k3=1;k3<=t;k3++){var p3=tip(k3);s+='<circle cx="'+p3[0].toFixed(1)+'" cy="'+p3[1].toFixed(1)+'" r="3.5" fill="#BA7517"/>';}
 var pc=tip(t);
 s+='<line x1="'+OX+'" y1="'+OY+'" x2="'+pc[0].toFixed(1)+'" y2="'+pc[1].toFixed(1)+'" stroke="#854F0B" stroke-width="1" stroke-dasharray="4 4"/>';
 s+='<line x1="'+pc[0].toFixed(1)+'" y1="'+pc[1].toFixed(1)+'" x2="'+pc[0].toFixed(1)+'" y2="'+OY+'" stroke="var(--text-muted)" stroke-width="0.5" stroke-dasharray="4 4"/>';
 s+='<circle cx="'+pc[0].toFixed(1)+'" cy="'+pc[1].toFixed(1)+'" r="6" fill="none" stroke="#854F0B" stroke-width="1.5"/>';
 s+='<text class="ts" x="'+(pc[0]+12).toFixed(1)+'" y="'+(pc[1]+(pc[1]<OY?-8:16)).toFixed(1)+'" fill="#854F0B">w after '+t+'</text>';
 s+='<text class="ts" x="'+(OX+Rr*0.82*Math.cos(th)).toFixed(1)+'" y="'+(OY-Rr*0.82*Math.sin(th)-6).toFixed(1)+'" text-anchor="middle" fill="#185FA5">cone at k='+t+'</text>';
 s+='<circle cx="'+OX+'" cy="'+OY+'" r="2.5" fill="var(--text-primary)"/>';
 document.getElementById('stage').innerHTML=s;
 document.getElementById('rd').innerHTML='shadow = <b>'+(t*g).toFixed(2)+'</b> = k&gamma; exactly &middot; |w| = <b>'+Math.sqrt(t).toFixed(2)+'</b> = &radic;k&middot;R exactly &middot; angle to w* <b>'+(th*180/Math.PI).toFixed(1)+'&deg;</b>, on the cone edge'+
  '<br>solution wedge <b>'+(wd*180/Math.PI).toFixed(1)+'&deg;</b> &middot; the zigzag reaches the axis and dies at k = R&sup2;/&gamma;&sup2; = <b>'+kb+'</b>';
}
function setg(v){g=v;document.getElementById('gv').textContent=v.toFixed(2);
 var sl=document.getElementById('ts');sl.max=KB();if(t>KB()){t=KB();sl.value=t;document.getElementById('tv').textContent=t;}render();}
function sett(v){t=v;document.getElementById('tv').textContent=v;render();}
setg(0.30);
