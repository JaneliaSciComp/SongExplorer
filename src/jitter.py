from natsort import natsorted, realsorted
import matplotlib.cm as cm
import statistics
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np

# data == { x1:[y1_1,y1_2,...], x2:[y2_1,y2_2,...], ... }
# xlabels = jitter_plot(ax, data)
# ax.set_xticks(range(len(xlabels)))
# ax.set_xticklabels(xlabels)

def jitter_plot(ax, data, orientation='vertical', reverse=False, \
                outlier_crit=0, outlier_offset=0.1, real=False):
  colors=[]; aveboxes=[]
  sortfun = realsorted if real else natsorted
  ldata = sortfun(data.keys(), reverse=reverse)
  for (i,l) in enumerate(ldata):
    colors.append(cm.viridis(i/max(1,len(ldata)-1)))
    x = ldata.index(l)
    d = [x for x in data[l] if x>outlier_crit]
    o = [x for x in data[l] if x<=outlier_crit]
    if len(d)<3:
      continue
    y = statistics.mean(d)
    h = statistics.stdev(d)
    if orientation=='vertical':
      aveboxes.append(Rectangle((x-0.25,y-h),0.5,2*h))
      ax.plot([x-0.25,x+0.25],[y,y],'w-')
    else:
      aveboxes.append(Rectangle((y-h,x-0.25),2*h,0.5))
      ax.plot([y,y],[x-0.25,x+0.25],'w-')

  pc = PatchCollection(aveboxes, facecolor='lightgrey')
  ax.add_collection(pc)

  xdata = [];  ydata = [];  cdata=[]
  for (i,k) in enumerate(ldata):
    kfold = len(data[k])
    if kfold>1:
      xdata.extend([i+float(x)/2/(kfold-1)-0.25 for x in range(len(data[k]))])
    else:
      xdata.extend([i])
    ydata.extend(data[k])
    cdata.extend([colors[i] for x in data[k]])

  xdata = np.array(xdata)
  ydata = np.array(ydata)
  cdata = np.array(cdata)
  ioutlier = np.where(ydata<=outlier_crit)
  iinlier = np.where(ydata>outlier_crit)
  if orientation=='vertical':
    if len(iinlier[0])>0:
      ax.scatter(xdata[iinlier], ydata[iinlier], c=cdata[iinlier], \
                 edgecolors='k', zorder=100)
    if len(ioutlier[0])>0:
      ax.scatter(xdata[ioutlier], \
                 np.full((len(ioutlier[0]),), np.min(ydata[iinlier])-outlier_offset), \
                 c=cdata[ioutlier],marker='v', zorder=100)
  else:
    if len(iinlier[0])>0:
      ax.scatter(ydata[iinlier], xdata[iinlier], c=cdata[iinlier], \
                 edgecolors='k', zorder=100)
    if len(ioutlier[0])>0:
      ax.scatter(np.full((len(ioutlier[0]),), np.min(ydata[iinlier])-outlier_offset), \
                 xdata[ioutlier], c=cdata[ioutlier], marker='v', zorder=100)

  return ldata
