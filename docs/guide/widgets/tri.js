// why a mistake barely lengthens w: the correction leans backward
var TH = 118, WL = 2.0;                    // angle in degrees between w and the correction, |w|
var OX = 170, OY = 250, S = 118;           // origin and scale in svg units
function P(x, y){ return [OX + S*x, OY - S*y]; }
function arrow(a, b, col, wd){ return '<line x1="'+a[0].toFixed(1)+'" y1="'+a[1].toFixed(1)+'" x2="'+
 b[0].toFixed(1)+'" y2="'+b[1].toFixed(1)+'" stroke="'+col+'" stroke-width="'+wd+
 '" marker-end="url(#arrow)"/>'; }
function render(){
 var t = TH * Math.PI / 180;
 var w = [WL, 0], c = [Math.cos(t), Math.sin(t)], n = [w[0]+c[0], w[1]+c[1]];
 var o = P(0,0), hw = P(w[0], w[1]), hn = P(n[0], n[1]);
 var s = '';
 // the right angle that bounds the growth, drawn as the reference case
 var rp = P(w[0], 1);
 s += '<line x1="'+hw[0].toFixed(1)+'" y1="'+hw[1].toFixed(1)+'" x2="'+rp[0].toFixed(1)+'" y2="'+
      rp[1].toFixed(1)+'" stroke="var(--border-strong)" stroke-width="1.2" stroke-dasharray="5 4"/>';
 s += '<line x1="'+o[0].toFixed(1)+'" y1="'+o[1].toFixed(1)+'" x2="'+rp[0].toFixed(1)+'" y2="'+
      rp[1].toFixed(1)+'" stroke="var(--border-strong)" stroke-width="1" stroke-dasharray="5 4"/>';
 s += '<text class="ts" x="'+(rp[0]+10).toFixed(1)+'" y="'+(rp[1]-6).toFixed(1)+
      '" fill="var(--text-muted)">the right-angle case: |w|&sup2; + 1</text>';
 s += arrow(o, hw, '#5F5E5A', 2.6);
 s += arrow(hw, hn, '#D85A30', 2.6);
 s += arrow(o, hn, '#1D9E75', 2.6);
 var a0 = P(w[0] + 0.34, 0), a1 = P(w[0] + 0.34*Math.cos(t), 0.34*Math.sin(t));
 s += '<path d="M '+a0[0].toFixed(1)+' '+a0[1].toFixed(1)+' A '+(0.34*S).toFixed(1)+' '+
      (0.34*S).toFixed(1)+' 0 0 0 '+a1[0].toFixed(1)+' '+a1[1].toFixed(1)+
      '" fill="none" stroke="var(--border-strong)" stroke-width="1.2"/>';
 s += '<text class="ts" x="'+(hw[0]+14).toFixed(1)+'" y="'+(hw[1]-46).toFixed(1)+
      '" fill="var(--text-secondary)">'+TH+'&deg;</text>';
 s += '<text class="ts" x="'+((o[0]+hw[0])/2).toFixed(1)+'" y="'+(hw[1]+22).toFixed(1)+
      '" text-anchor="middle" fill="var(--text-primary)">w</text>';
 s += '<text class="ts" x="'+((hw[0]+hn[0])/2+10).toFixed(1)+'" y="'+((hw[1]+hn[1])/2).toFixed(1)+
      '" fill="#9c3d17">y&middot;x</text>';
 s += '<text class="ts" x="'+((o[0]+hn[0])/2-34).toFixed(1)+'" y="'+((o[1]+hn[1])/2).toFixed(1)+
      '" fill="#0F6E56">w + y&middot;x</text>';
 document.getElementById('stage').innerHTML = s;
 var got = n[0]*n[0] + n[1]*n[1], cap = WL*WL + 1, mis = TH >= 90;
 document.getElementById('readout').innerHTML =
  '|w + y&middot;x|&sup2; = <b>'+got.toFixed(2)+'</b> against |w|&sup2; + |x|&sup2; = <b>'+cap.toFixed(2)+'</b>'+
  ' &middot; ' + (mis
   ? 'the angle is obtuse, which <i>is</i> the mistake condition, so the growth stays under the cap.'
   : 'below 90&deg; the point was already correct, the correction leans forward, and the cap breaks.');
}
function sett(v){ TH = v; document.getElementById('tv').textContent = v + '°'; render(); }
sett(118);
