// the two gauges colliding, with gamma on a slider
var G = 0.35, KMAX = 20, VMAX = 5;
var L = 70, R = 580, T = 45, B = 285;
function X(k){ return L + (k / KMAX) * (R - L); }
function Y(v){ return B - (Math.min(v, VMAX) / VMAX) * (B - T); }
function render(){
 var s = '', i;
 s += '<line x1="'+L+'" y1="'+T+'" x2="'+L+'" y2="'+B+'" stroke="var(--border-strong)" stroke-width="1"/>';
 s += '<line x1="'+L+'" y1="'+B+'" x2="'+R+'" y2="'+B+'" stroke="var(--border-strong)" stroke-width="1"/>';
 var pts = '';
 for (i = 0; i <= 120; i++){ var k = KMAX * i / 120; pts += X(k).toFixed(1)+','+Y(Math.sqrt(k)).toFixed(1)+' '; }
 s += '<polyline points="'+pts+'" fill="none" stroke="#378ADD" stroke-width="2.4"/>';
 var kend = Math.min(KMAX, VMAX / G);
 s += '<line x1="'+L+'" y1="'+B+'" x2="'+X(kend).toFixed(1)+'" y2="'+Y(kend*G).toFixed(1)+
      '" stroke="#D85A30" stroke-width="2.4"/>';
 var kx = 1 / (G*G);
 if (kx <= KMAX){
  s += '<line x1="'+X(kx).toFixed(1)+'" y1="'+Y(kx*G).toFixed(1)+'" x2="'+X(kx).toFixed(1)+'" y2="'+B+
       '" stroke="var(--border-strong)" stroke-width="1" stroke-dasharray="4 4"/>';
  s += '<circle cx="'+X(kx).toFixed(1)+'" cy="'+Y(kx*G).toFixed(1)+'" r="5" fill="var(--text-primary)"/>';
  s += '<text class="ts" x="'+X(kx).toFixed(1)+'" y="'+(B+20)+'" text-anchor="middle" '+
       'fill="var(--text-secondary)">k = 1/&gamma;&sup2; = '+kx.toFixed(1)+'</text>';
 } else {
  s += '<text class="ts" x="'+(R-6)+'" y="'+(B+20)+'" text-anchor="end" fill="var(--text-muted)">'+
       'crossing at k = '+kx.toFixed(0)+', off the right of the plot</text>';
 }
 s += '<text class="ts" x="'+X(KMAX*0.62).toFixed(1)+'" y="'+Math.max(T+14, Y(KMAX*0.62*G)-10).toFixed(1)+
      '" text-anchor="middle" fill="#9c3d17">shadow on true north = k&gamma;</text>';
 s += '<text class="ts" x="'+X(KMAX*0.30).toFixed(1)+'" y="'+(Y(Math.sqrt(KMAX*0.30))-16).toFixed(1)+
      '" text-anchor="middle" fill="#1f5f9e">length of w = &radic;k</text>';
 s += '<text class="ts" x="'+((L+R)/2)+'" y="'+(B+42)+'" text-anchor="middle" fill="var(--text-muted)">mistakes k</text>';
 document.getElementById('stage').innerHTML = s;
 document.getElementById('readout').innerHTML =
  'margin <b>&gamma; = '+G.toFixed(2)+'</b> &middot; the straight line overtakes the curve at '+
  '<b>k = 1/&gamma;&sup2; = '+kx.toFixed(1)+'</b>, so there are no mistakes left after that.';
}
function setg(v){ G = v; document.getElementById('gv').textContent = v.toFixed(2); render(); }
setg(0.35);
