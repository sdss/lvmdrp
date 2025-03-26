#!/usr/bin/python3
import numpy as np
import sys
from astropy import units as u
from skyfield.api import Loader
from skyfield.api import Topos

from lvmdrp.core.constants import EPHEMERIS_DIR

class shadow_calc(object):
    def __init__(self, observatory_name="LCO",
    observatory_elevation=2380.0*u.m,
    observatory_lat='29.0146S',
    observatory_lon = '70.6926W',
    jd=2459458,
    eph=None,
    earth=None,
    sun=None):
        """
        Initialization sets a default observatory to LCO, and the default time to the date when initialized.
        """
        super().__init__()

        # Load the ephemeral datat for the earth and sun.
        load = Loader(EPHEMERIS_DIR)
        if eph is None:
            self.eph = load('de421.bsp')

        # Get functions for the earth, sun and observatory
        self.earth = earth
        if self.earth is None:
            self.earth = self.eph['earth']

        self.sun = sun
        if self.sun is None:
            self.sun = self.eph['sun']

        self.observatory_elevation = observatory_elevation
        try:
            self.observatory_elevation.to("m")
        except (AttributeError, u.UnitConversionError):
            sys.exit("Observatory elevation does not have unit of length")

        self.observatory_name = observatory_name
        self.observatory_lat = observatory_lat
        self.observatory_lon = observatory_lon
        self.observatory_topo = Topos(observatory_lat, observatory_lon, elevation_m=self.observatory_elevation.to("m").value)
        self.ts = load.timescale()

        """
        These are going to be vectors which are the x,y,z positions of the puzzle.
        Credit for this solution to the intersection goes to:  Julien Guertault @ http://lousodrome.net/blog/light/2017/01/03/intersection-of-a-ray-and-a-cone/
        """
        self.ra = None
        self.dec = None
        self.pointing_unit_vectors = None
        self.xyz_earth = None
        self.xyz_sun = None
        self.xyz_observatory = None
        self.v = None
        self.xyz_c = None
        self.a = None
        self.b = None
        self.c = None
        self.delta = None
        self.dist = None
        self.heights = None

        # Those are very self explanitory
        self.earth_radius=6.357e6*u.m
        self.sun_radius=695.700e6*u.m

        # Distance from Sun to the earth.
        self.d_se = 1*u.au

        # Distance from tip of the shadow cone to the earth
        self.d_ec = self.d_se * (self.earth_radius / self.sun_radius)

        # Opening angle of the shadow cone
        self.shadow_cone_theta  = np.arctan(self.earth_radius/self.d_ec)
        self.shadow_cone_cos_theta_sqr = np.square(np.cos(self.shadow_cone_theta))

        # Finally, updated the xyz coordinate vectors of earth, sun and observatory.
        self.jd = jd
        if jd is not None:
            self.t = self.ts.tt_jd(jd)
        else:
            self.t = self.ts.now()

        self.update_time(jd)

    def cone_ra_dec(self):
        # This returns the cone RA,Dec for testing.
        # It has a shadow height that is analyically easy to calculate
        observatory_to_c = self.xyz_c - self.xyz_observatory

        # Leave dec in steradian for just a moment
        dec = np.arctan(observatory_to_c[2]/np.sqrt(np.square(observatory_to_c[0])+np.square(observatory_to_c[1])))

        # Use dec in steradians to solve for ra.
        ra = np.degrees(np.arccos(observatory_to_c[0]/ (self.vecmag(observatory_to_c) * np.cos(dec))))

        #Now convert to degree
        dec = np.degrees(dec)

        return ra, dec

    def set_coordinates(self, ra, dec):
        # Set coordinates and calculate unit vectors.
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        self.ra = ra
        self.dec = dec
        self.pointing_unit_vectors = np.zeros(shape=(len(self.ra),3))

        a = np.pi / 180.
        self.pointing_unit_vectors[:,0] = np.cos(self.ra*a)*np.cos(self.dec*a)
        self.pointing_unit_vectors[:,1] = np.sin(self.ra*a)*np.cos(self.dec*a)
        self.pointing_unit_vectors[:,2] = np.sin(self.dec*a)

    def update_time(self, jd=None):
        """ Update the time, and all time dependent vectors """
        if jd is not None:
            self.jd = jd
        self.t = self.ts.tt_jd(self.jd)
        self.update_xyz()
        self.update_vec_c()
        self.co = self.xyz_c - self.xyz_observatory

    def update_xyz(self):
        """
        Due to the nature of the vectors, it is not possible to combine the earth and the observatory topo directly. Therefore
        this requires the calculation of an observatory position to be done using the XYZ positions defined at a particular time, and with a particular unit.
        To prevent unit issues later from arising SI units are adopted. In the case where other units are desired, like plotting the solar system, local conversions
        can be done.
        """
        self.xyz_earth = self.earth.at(self.t).position.au
        self.xyz_sun = self.sun.at(self.t).position.au
        self.xyz_observatory = self.earth.at(self.t).position.au + self.observatory_topo.at(self.t).position.au


    def update_vec_c(self):
        """
        d_se : distance sun to earth
        d_ec : distance earth to (c)one tip
        c_unit_vec: unit vector from cone to earth, or earth to sun.

        """
        # We reverse the vector sun to earth explicitly with -1.0 for clarity
        self.v = -1.0*(self.xyz_earth - self.xyz_sun)/np.sqrt(np.sum(np.square(self.xyz_earth-self.xyz_sun)))
        self.xyz_c = self.xyz_earth - self.v * self.d_ec.to("au").value

    def solve_for_height(self, mask=None, unit="km"):
        if mask is None:
            mask = np.full(len(self.pointing_unit_vectors), True)

        puv = self.pointing_unit_vectors[mask]

        self.a = np.square(np.sum(puv*self.v,axis=1)) -  self.shadow_cone_cos_theta_sqr
        self.b = 2*( np.sum(puv*self.v, axis=1) * np.sum(self.co*self.v) - np.sum(puv*self.co, axis=1)*self.shadow_cone_cos_theta_sqr)
        self.c = np.square(np.sum(self.co*self.v)) - np.sum(self.co * self.co) * self.shadow_cone_cos_theta_sqr
        self.delta = np.square(self.b) - 4 * self.a * self.c

        self.dist = np.full(len(self.a), np.nan)
        dist_b1 = np.full(len(self.a), np.nan)
        dist_b2 = np.full(len(self.a), np.nan)

        # Get P distance to point.
        positive_delta = self.delta >= 0
        dist_b1[positive_delta] = -1*(-self.b[positive_delta]+np.sqrt(self.delta[positive_delta])) / (2*self.a[positive_delta])
        dist_b2[positive_delta] = -1*(-self.b[positive_delta]-np.sqrt(self.delta[positive_delta])) / (2*self.a[positive_delta])

        dist_b1[dist_b1 < 0.0] = np.nan
        dist_b2[dist_b2 < 0.0] = np.nan

        self.dist[positive_delta] = np.nanmin([dist_b1[positive_delta], dist_b2[positive_delta]], axis=0)

        # caluclate xyz of each ra, using the distance to the shadow intersection and the normal vector form the observatory
        pointing_xyz = puv * self.dist[:,np.newaxis] + self.xyz_observatory

        self.heights = (self.vecmag(pointing_xyz - self.xyz_earth)*u.au - self.earth_radius).to(unit).value

    def get_heights(self, mask=None, jd=None, return_heights=True,unit=u.km):
        if jd is not None:
            self.jd = jd
            self.update_time()
        self.solve_for_height(mask=mask, unit=unit)
        if return_heights:
            return self.heights

    def vecmag(self, a):
        """ Return the magnitude of a set of vectors around an abritrary origin """
        if len(np.shape(a)) == 1:
            return np.sqrt(np.square(a[0]) + np.square(a[1]) + np.square(a[2]))
        else:
            return np.sqrt(np.square(a[:, 0]) + np.square(a[:, 1]) + np.square(a[:, 2]))

class orbit_animation(object):
    def __init__(self, calculator, jd0=None, djd=1/24.):
        if jd0 is None:
            self.jd0 = calculator.jd
        else:
            self.jd0 = jd0
        self.djd = djd

        self.calculator = calculator
        self.calculator.observatory_zenith_topo = Topos(calculator.observatory_lat, calculator.observatory_lon, elevation_m=self.calculator.earth_radius.to("m").value*10)

        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.patches as patches

        self.plt = plt
        self.animation = animation
        self.patches = patches
        self.fig, self.ax = self.plt.subplots(figsize=(50,50))

    def init_plotting_animation(self):
        self.line1, = self.ax.plot([], [], 'ro', lw=2)
        self.line2, = self.ax.plot([], [], '.-', lw=5)
        self.line3, = self.ax.plot([], [], '.-', lw=5)
        self.path, = self.ax.plot([],[],'-', lw=1)

        self.ax.grid()
        self.xdata, self.ydata = [], []
        self.pathx, self.pathy = [], []



        self.ax.set_ylim(-150 * self.calculator.earth_radius.to("au").value, 150 * self.calculator.earth_radius.to("au").value)
        self.ax.set_xlim(-150 * self.calculator.earth_radius.to("au").value, 150 * self.calculator.earth_radius.to("au").value)
        del self.xdata[:]
        del self.ydata[:]
        self.line1.set_data(self.xdata, self.ydata)
        self.line2.set_data(self.xdata, self.ydata)

    def do_animation(self, N_days=1.0):
        ani = self.animation.FuncAnimation(self.fig, self.animation_update_positions, frames=(24*N_days), repeat_delay=0,
                                        blit=False, interval=100, init_func=self.init_plotting_animation, repeat=0)
        ani.save("/tmp/im.mp4")

    def snap_shot(self,jd,ra,dec, show=True, extra=False):
        self.init_plotting_animation()
        self.calculator.set_coordinates(np.array([ra]),np.array([dec]))

        self.calculator.update_time(jd)

        self.calculator.xyz_observatory_zenith = self.calculator.xyz_earth + self.calculator.observatory_zenith_topo.at(self.calculator.t).position.au

        height_au = 3.0 * self.calculator.earth_radius.to("au").value

        self.ax.set_xlim( ( self.calculator.xyz_earth[0] - height_au * 160, self.calculator.xyz_earth[0] + height_au * 160 ) )
        self.ax.set_ylim( ( self.calculator.xyz_earth[1] - height_au * 90, self.calculator.xyz_earth[1] + height_au * 90 ) )

        circ = self.patches.Circle((self.calculator.xyz_earth[0], self.calculator.xyz_earth[1]), self.calculator.earth_radius.to( "au" ).value, alpha=0.8, fc='yellow')
        self.ax.add_patch( circ )

        los_xyz =  self.calculator.xyz_observatory + self.calculator.d_ec.to("au").value * self.calculator.pointing_unit_vectors[0]

        #self.line1.set_data( (x_heights * u.m).to( "au" ).value, ( y_heights * u.m ).to("au").value )
        self.line1.set_data( [ self.calculator.xyz_c[0] ], [ self.calculator.xyz_c[1] ] )
        self.line2.set_data( [ self.calculator.xyz_observatory_zenith[0] ], [ self.calculator.xyz_observatory_zenith[1] ] )
        self.line3.set_data( [ self.calculator.xyz_observatory[0] ] , [ self.calculator.xyz_observatory[1] ] )
        self.ax.plot([self.calculator.xyz_c[0], self.calculator.xyz_sun[0]],[self.calculator.xyz_c[1], self.calculator.xyz_sun[1]], alpha=0.5)
        self.ax.plot([los_xyz[0], self.calculator.xyz_observatory[0]],[los_xyz[1], self.calculator.xyz_observatory[1]], c="k", alpha=0.5)
        if extra is not False:
            #This will plot a line using arrays provided in the extra bin.
            self.ax.plot(extra[0], extra[1])

        self.plt.gca().set_aspect('equal', adjustable='box')
        if show:
            self.plt.show()
        else:
            return self.plt


    def animation_update_positions(self, frame):
        if len( self.ax.patches ) > 1:
            del( self.ax.patches[-1] )

        jd = self.jd0 + frame*self.djd

        #self.calculator.t = self.calculator.ts.tt_jd( self.jd0 + frame*self.djd)
        self.calculator.update_time(jd)

        self.calculator.xyz_observatory_zenith = self.calculator.xyz_earth + self.calculator.observatory_zenith_topo.at(self.calculator.t).position.au

        height_au = 3.0 * self.calculator.earth_radius.to("au").value

        ######################################################################
        # The next set of lines calculates shadow heights. That's for later  #
        ######################################################################
        # N_ra = 24
        # N_dec = 9
        # x_heights = np.zeros(len(N_ra * N_dec)+1)
        # y_heights = np.zeros(len(N_ra * N_dec)+1)
        #
        # for ra_i, ra in enumerate(np.linspace(1,24,N_ra)):
        #     for dec_j, dec in enumerate(np.linspace(-90, 0, num=N_dec, endpoint=True)):
        #         # for height in heights:
        #             self.calculator.point_along_ray =  position_from_radec(ra, dec, distance=height_au, epoch=None, t=self.calculator.t, center=self.calculator.observatory_topo, target=None) #, observer_data=LCO_topo.at(t).observer_data)
        #             self.calculator.xyz_along_ray = self.calculator.xyz_observatory + self.calculator.point_along_ray.position.au
        #             x_heights[ ra_i + dec_j*N_ra ] = self.calculator.xyz_along_ray[0]
        #             y_heights[ ra_i + dec_j*N_ra ] = self.calculator.xyz_along_ray[1]
        #
        # x_heights[-1] = self.calculator.xyz_earth[0]
        # y_heights[-1] = self.calculator.xyz_earth[1]
        ######################################################################
        self.ax.set_xlim( ( self.calculator.xyz_earth[0] - height_au * 150, self.calculator.xyz_earth[0] + height_au * 150 ) )
        self.ax.set_ylim( ( self.calculator.xyz_earth[1] - height_au * 150, self.calculator.xyz_earth[1] + height_au * 150 ) )

        circ = self.patches.Circle((self.calculator.xyz_earth[0], self.calculator.xyz_earth[1]), self.calculator.earth_radius.to( "au" ).value, alpha=0.8, fc='yellow')
        self.ax.add_patch( circ )

        #self.line1.set_data( (x_heights * u.m).to( "au" ).value, ( y_heights * u.m ).to("au").value )
        self.line1.set_data( [ self.calculator.xyz_c[0] ], [ self.calculator.xyz_c[1] ] )
        self.line2.set_data( [ self.calculator.xyz_observatory_zenith[0] ], [ self.calculator.xyz_observatory_zenith[1] ] )
        self.line3.set_data( [ self.calculator.xyz_observatory[0] ] , [ self.calculator.xyz_observatory[1] ] )
        self.pathx.append( self.calculator.xyz_earth[0] )
        self.pathy.append( self.calculator.xyz_earth[1] )
        self.path.set_data( self.pathx, self.pathy )

        #line2.set_data([sun.at(t).position.au[0]],[sun.at(t).position.au[1]])

def test_shadow_calc():
    jd = 2459458.5+20 + (4.75)/24.
    # Initiate tests.
    test_results ={}

    try:
        test = "Create Class"
        calculator = shadow_calc()
        test_results[test] = "Pass"
    except Exception:
        test_results[test] = "Fail"

    orbit_ani = orbit_animation(calculator)
    orbit_ani.calculator.update_time(jd)
    ra, dec = orbit_ani.calculator.cone_ra_dec()

    orbit_ani.snap_shot(jd=jd, ra=ra, dec=dec)
    # orbit_ani.do_animation()

    compare_old = True
    if compare_old:
        load = Loader(EPHEMERIS_DIR)
        eph = load('de421.bsp')
        import lvmsurveysim.utils.iterative_shadow_height_lib as iterative_shadow_height_lib
        iter_calc = iterative_shadow_height_lib.shadow_calc(observatory_name='LCO',
                        observatory_elevation=2380*u.m,
                        observatory_lat='29.0146S', observatory_lon='70.6926W',
                        eph=eph, earth=eph['earth'], sun=eph['sun'])
        iter_calc.update_t(jd)
        old_h = iter_calc.height_from_radec(ra/15., dec, simple_output=True)['height']
        new_h = calculator.get_heights(return_heights=True, unit="m")
        print("old: %f; new: %f"%(old_h, new_h))

    try:
        test = "Shadow RA/DEC"
        ra, dec = calculator.cone_ra_dec()
    except Exception:
        test_results[test] = "Fail"

    try:
        test = "Set Coordinates"
        calculator.set_coordinates(np.array([ra]), np.array([dec]) )
        test_results[test] = "Pass"
    except Exception:
        test_results[test] = "Fail"

    try:
        test = "set jd=2459458.5"
        jd = 2459458.5+20 + 4.75/24.
        calculator.update_time(jd)
        test_results[test] = "Pass"
    except Exception:
        test_results[test] = "Fail"

    try:
        test = "Run"
        calculator.get_heights()
        test_results[test] = "Pass"
    except Exception:
        test_results[test] = "Fail"

    try:
        test = "loop"
        for jd in np.linspace(jd, jd+1, 1/24.):
            calculator.update_time(jd)
            # heights = calculator.get_heights(return_heights=True, unit="km")
            # print("height_v(jd=%f) = %f"%(jd, heights[0]))
        test_results[test] = "Pass"
    except Exception:
        test_results[test] = "Fail"

    for test in test_results.keys():
        print ("%s -> %s"%(test, test_results[test]))


if __name__ == "__main__":
    test_shadow_calc()