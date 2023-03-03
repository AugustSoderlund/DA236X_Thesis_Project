from pprint import pprint
import osmium as osm
import math
import pyproj
import matplotlib.pyplot as plt
import re
import numpy as np
import alphashape
from descartes import PolygonPatch
from shapely.geometry import LineString

ROOT = "C:/Users/ASOWXU/Desktop/Thesis Project/Code/DA236X_Thesis_Project/thesis_project/.datasets/SinD/Data/mapfile-Tianjin.osm"

class SinD_map:
    """ Map class for the SinD dataset
    
        Parameters:
        -----------
        map_dir : str
            The absolute (or relative) path to the location where the .osm file is located

        Functions:
        ----------
        *** ADD FUNCTIONS ***
    """
    def __init__(self, map_dir: str = ROOT):
        self.osm_data = self.__load_osm_data(map_dir)
        self.__intersection_polygon(), self.__crosswalk_polygon(), self.__gap_polygon(), self.__road_and_sidewalk_polygon()
        self.__fix_poly_sizes()

    def __load_osm_data(self, map_dir: str):
        osmhandler = OSMHandler()
        osmhandler.apply_file(map_dir)
        return osmhandler.osm_data
    
    def plot_areas(self):
        _, ax = plt.subplots()
        __points = self.get_area("")
        ax.scatter(*zip(*__points), alpha=0) # To get bounds correct
        ax.add_patch(PolygonPatch(self.crosswalk_poly, alpha=0.2, color="b"))
        ax.add_patch(PolygonPatch(self.intersection_poly, alpha=0.2, color="r"))
        ax.add_patch(PolygonPatch(self.gap_poly, alpha=0.2, color="r"))
        ax.add_patch(PolygonPatch(self.road_poly, alpha=0.2, color="r"))
        ax.add_patch(PolygonPatch(self.sidewalk_poly, alpha=0.2, color="g"))
        plt.show()
    
    def get_area(self, regex: str = "crosswalk"):
        __ways, __nodes, __locs = [], [], []
        for (_, values) in self.osm_data["Relations"].items():
            tags = values["tags"]
            if "name" not in tags.keys():
                break
            __found = re.findall(regex, tags["name"])
            if __found:
                __ways = [*__ways, *values["way_members"]]
        for __way in __ways:
            __nodes = [*__nodes, *self.osm_data["Ways"][__way]["nodes"]]
        for __node in __nodes:
            __locs.append(self.osm_data["Nodes"][__node])
        return __locs
    
    def __crosswalk_polygon(self):
        __points = self.get_area("crosswalk")
        crosswalk_shape = self.__get_exterior(__points)
        self.crosswalk_poly = crosswalk_shape.difference(self.intersection_poly) if re.findall("Tianjin", ROOT) else crosswalk_shape

    def __intersection_polygon(self):
        __points = self.get_area("inter")
        self.intersection_poly = self.__get_exterior(__points)
    
    def __gap_polygon(self):
        __points = self.get_area("gap")
        self.gap_poly = self.__get_exterior(__points)
    
    def __road_and_sidewalk_polygon(self, sidewalk_size: float = 8.0):
        __points = self.get_area("")
        cross_poly = self.crosswalk_poly
        inter_poly = self.intersection_poly
        gap_poly = self.gap_poly
        road_poly = self.__get_exterior(__points)
        sidewalk_poly = road_poly.buffer(sidewalk_size).difference(road_poly).intersection(road_poly.minimum_rotated_rectangle.buffer(-0.2))
        self.road_poly, self.sidewalk_poly = road_poly.difference(cross_poly).difference(inter_poly).difference(gap_poly), sidewalk_poly

    def __fix_poly_sizes(self):
        self.intersection_poly = self.intersection_poly.difference(self.crosswalk_poly.buffer(3)).buffer(3)
        self.crosswalk_poly = self.crosswalk_poly.buffer(3).difference(self.intersection_poly).difference(self.road_poly).difference(self.sidewalk_poly).difference(self.gap_poly)

    def __get_exterior(self, points: list, alpha: float = 0.4):
        return alphashape.alphashape(np.array(points), alpha=alpha)
        


class OSMHandler(osm.SimpleHandler):
    """ OpenStreetMap handler that reads the nodes, ways and 
        relations from a .osm-file
    """
    def __init__(self):
        """ Format for osm_data 
                osm_data = {
                    Nodes: {
                        _id: [x, y]
                    },
                    Ways" {
                        _id: {
                            nodes: [_id1, ..., _idn],
                            tags: {_tags}
                        }
                    },
                    Relations: {
                        _id: {
                            way_members: [_id1, _id2],
                            tags: {_tags}
                        }
                    }
                }
        """
        osm.SimpleHandler.__init__(self)
        self.projector = LL2XYProjector()
        self.osm_data = {"Nodes": {}, "Ways": {}, "Relations": {}}

    def node(self, n):
        [x, y] = self.projector.latlon2xy(n.location.lat, n.location.lon)
        self.osm_data["Nodes"].update({n.id: [x, y]})

    def way(self, w):
        self.osm_data["Ways"].update({w.id: {"nodes": [n.ref for n in w.nodes], "tags": dict(w.tags)}})

    def relation(self, r):
        self.osm_data["Relations"].update({r.id: {"way_members": [m.ref for m in r.members if m.type == "w"], "tags": dict(r.tags)}})


class LL2XYProjector:
    """ Projector class that projects longitude and latitude
        onto the xy-plane.

        Parameters:
        -----------
        lat_origin : float
            origin for latitude (default: 0)
        lon_origin : float
            origin for longitude (default: 0)

        Functions:
        ----------
        latlon2xy(lat: float, lon: float)
            converts latitude and longitude to xy-coordinates 
            given the lat- and lon-origin
    """
    def __init__(self, lat_origin: float = 0, lon_origin: float = 0):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)
    
    def latlon2xy(self, lat: float, lon: float):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]



if __name__ == "__main__":
    map = SinD_map()
    map.plot_areas()