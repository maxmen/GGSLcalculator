import numpy as np
import astropy.io.fits as pyfits
import shapely.geometry
from shapely.ops import polygonize, unary_union
from astropy.cosmology import FlatLambdaCDM
from skimage import measure
from scipy.ndimage import map_coordinates
import argparse

class deflector(object):

    def __init__(self,co,angx=None,angy=None,**kwargs):

        if ('zl' in kwargs):
            self.zl=kwargs['zl']
        else:
            self.zl=0.5
        if ('zs' in kwargs):
            self.zs=kwargs['zs']
        else:
            self.zs=1.0

        self.angx=angx
        self.angy=angy
        self.co=co
        self.dl = self.co.angular_diameter_distance(self.zl)
        self.ds = self.co.angular_diameter_distance(self.zs)
        self.dls = self.co.angular_diameter_distance_z1z2(self.zl, self.zs)
        self.a21,self.a11=np.gradient(self.angx)
        self.a22,self.a12=np.gradient(self.angy)

    def angle(self,theta1,theta2):
        return (self.angx, self.angy)

    def kappa(self, theta1, theta2):
        dtheta = np.abs(theta1[0,1]-theta1[0,0])
        return (0.5 * (self.a11 / dtheta + self.a22 / dtheta))

    def gamma(self, theta1, theta2):
        dtheta = np.abs(theta1[0,1]-theta1[0,0])
        return (0.5 * (self.a11 - self.a22) / dtheta, self.a12 / dtheta)

    def setGrid(self,theta):
        self.theta1, self.theta2 = np.meshgrid(theta, theta)
        self.theta=theta

        self.grid_pixel=self.theta[1]-self.theta[0]

        self.size=np.max(self.theta)-np.min(self.theta)
        self.g1, self.g2 = self.gamma(self.theta1, self.theta2)
        self.ka = self.kappa(self.theta1, self.theta2)
        self.a1,self.a2=self.angle(self.theta1, self.theta2)

        self.nray = len(theta)
        self.pixel_scale = self.theta[1]-self.theta[0]

    
    def tancl(self,size_principale=5.0):
        lambdat=1.0-self.ka-np.sqrt(self.g1*self.g1+self.g2*self.g2)
        contour_ = measure.find_contours(lambdat, 0.0)
        contour = contour_[0].copy()
        contour[:, 0], contour[:, 1] = contour_[0][:, 1].copy(), contour_[0][:, 0].copy()
        ls = shapely.geometry.LineString(contour)
        lr = shapely.geometry.LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(lr)
        mp = shapely.geometry.MultiPolygon(list(polygonize(mls)))
        lc = CriticalLine(0, mp)
        lc.setPoints(contour)
        A = 0.0
        for g in range(len(mp)):
            A += mp[g].area
        lc.setArea(A)
        if lc.getThetaE()*self.grid_pixel > size_principale:
            lc.principale = True
        return(lc)

    def getCritPoints(self,lc,pixel_units=False):

        vs=lc.points
        x1,x2=zip(*vs)
        if (pixel_units):
            return(np.array(x1),np.array(x2))
        else:
            x1=(np.array(x1)-self.a1.shape[0]/2.0)*self.grid_pixel
            x2=(np.array(x2)-self.a1.shape[0]/2.0)*self.grid_pixel
            return(np.array(x1),np.array(x2))

    def mapCrit2Cau(self, x1, x2):

        a1 = map_coordinates(self.a1, [[x2], [x1]], order=3, prefilter=True)
        a2 = map_coordinates(self.a2, [[x2], [x1]], order=3, prefilter=True)
        y1 = x1 - a1[0]/self.grid_pixel+1
        y2 = x2 - a2[0]/self.grid_pixel+1
        return (y1, y2)

    def getCaustics(self,lc_ord):
        x1,x2=self.getCritPoints(lc_ord,pixel_units=True)
        y1,y2=self.mapCrit2Cau(x1,x2)
        if (y1.size>1):
            points=list(zip(y1,y2))
            points=np.asarray(points)
        else:
            points=[(0, 0), (0, 0)]
        ls = shapely.geometry.LineString(points)
        lr = shapely.geometry.LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(lr)
        mp = shapely.geometry.MultiPolygon(list(polygonize(mls)))
        cau=Caustic(lc_ord.ID,mp)
        cau.principale=lc_ord.principale
        cau.passsize=lc_ord.passsize
        cau.setPoints(points)
        A=mp.area
        cau.setArea(A)
        return(cau)

    def getCausticPoints(self,cau,pixel_units=False):
        vs=cau.points
        y1,y2=zip(*vs)
        if (pixel_units):
            return(y1,y2)
        else:
            y1=(np.array(y1)-self.a1.shape[0]/2.0)*self.grid_pixel
            y2=(np.array(y2)-self.a1.shape[0]/2.0)*self.grid_pixel
        return(y1,y2)

    def change_redshift(self,newzs):
        ds_ = self.co.angular_diameter_distance(newzs)
        dls_ = self.co.angular_diameter_distance_z1z2(self.zl, newzs)
        self.ka=self.ka*self.ds.value/self.dls.value*dls_.value/ds_.value
        self.g1=self.g1*self.ds.value/self.dls.value*dls_.value/ds_.value
        self.g2=self.g2*self.ds.value/self.dls.value*dls_.value/ds_.value
        self.a1=self.a1*self.ds.value/self.dls.value*dls_.value/ds_.value
        self.a2=self.a2*self.ds.value/self.dls.value*dls_.value/ds_.value
        self.zs=newzs
        self.ds=ds_
        self.dls=dls_

    def ggslCrossSection(self,minsize=0.5,maxsize=5.0,dmax=200.0):
        clt = self.tancl()
        dist = np.sqrt((clt.px-self.nray/2.0)**2+(clt.py-self.nray/2.0)**2)*self.grid_pixel
        caut = self.getCaustics(clt)
        area = 0.0
        
        print (clt.getThetaE()*self.grid_pixel)
        if (dist < dmax) & \
            (clt.getThetaE()*self.grid_pixel > minsize) & \
            (clt.getThetaE()*self.grid_pixel < maxsize) & \
            (not clt.principale):
            area = caut.getArea()
        return area*self.grid_pixel**2


class CriticalLine(object):

    def __init__(self, ID, geometria):
        self.ID = int(ID)
        self.principale = False
        self.passsize = False
        self.geometria = geometria
        self.sigmaloc = 10000.0

    def setArea(self, area):
        self.area = area

    def setPoints(self, points):
        self.points = points
        c1, c2 = zip(*points)
        c1 = np.array(c1)
        c2 = np.array(c2)
        self.px = c1.mean()
        self.py = c2.mean()

    def getArea(self):
        return self.area

    def getThetaE(self):
        return np.sqrt(self.area/np.pi)

class Caustic(object):

    def __init__(self, ID, geometria):
        self.ID = ID
        self.geometria = geometria
        self.principale = False
        self.passsize = False
        self.sigmaloc = 10000.0

    def setArea(self, area):
        self.area = area

    def setPoints(self, points):
        self.points = points

    def getArea(self):
        return self.area

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = "input filename")
    args = parser.parse_args()

    hdul=pyfits.open(args.input)
    a1=hdul[0].data
    a2=hdul[1].data

    co=FlatLambdaCDM(Om0=0.3,H0=70)
    kwargs_def={'zl': 0.3, 'zs': 3.0}

    theta=np.linspace(-100,100,2048)
    df=deflector(co,angx=a1,angy=a2,**kwargs_def)
    df.setGrid(theta=theta)

    zs_arr=np.linspace(0.7,7.0,10)
    for zs_ in zs_arr:
        df.change_redshift(zs_)
        print ("Source redshift: %f Cross section: %f" % (zs_,df.ggslCrossSection()))