from pprint import pprint
import osmium as osm
import math
import pyproj
import matplotlib.pyplot as plt
import re
import numpy as np
import alphashape
from descartes import PolygonPatch

ROOT = "C:/Users/ASOWXU/Desktop/Thesis Project/Code/DA236X_Thesis_Project/thesis_project/.datasets/"

class SinD_map:
    """ Map class for the SinD dataset
    
        Parameters:
        -----------
        map_dir : str
            The absolute (or relative) path to the location where the .osm file is located

        Attributes:
        -----------
        osm_data            : dict
        croswalk_poly       : shapely.geometry.multipolygon.MultiPolygon
        intersection_poly   : shapely.geometry.multipolygon.MultiPolygon
        gap_poly            : shapely.geometry.multipolygon.MultiPolygon
        road_poly           : shapely.geometry.multipolygon.MultiPolygon
        sidewalk_poly       : shapely.geometry.multipolygon.MultiPolygon
        
        Functions:
        ----------
        plot_areas(alpha: float) -> matplotlib.pyplot.Axes
            Gets all the areas and plots the PolygonPatches in a matplotlib figure 
            with the transparency alpha

        get_area(regex: str, tag_key: str) -> list
            given a regular expression and a key in the tags-list of the data-dict,
            this function returns the location of all nodes beloning to the region
            given by the regex

    """
    def __init__(self, map_dir: str = "SinD/Data/mapfile-Tianjin.osm"):
        self._map_dir = map_dir
        self.osm_data = self.__load_osm_data(ROOT + map_dir)
        self.__initialize_polygons() if re.findall("SinD", map_dir) else None

    def __load_osm_data(self, map_dir: str):
        osmhandler = OSMHandler()
        osmhandler.apply_file(map_dir)
        return osmhandler.osm_data
    
    def plot_areas(self, highlight_areas: list = ["crosswalk", "sidewalk"], alpha: float = 0.08):
        fig, ax = plt.subplots()
        fig.set_size_inches(6.5, 4.13)
        fig.subplots_adjust(top=0.95, left=0.08, bottom=0.1, right=0.95)
        _points = self.get_area("")
        ax.scatter(*zip(*_points), alpha=0) # To get bounds correct
        _attr = dir(self)
        _polys = [v for (_, v) in enumerate(_attr) if re.findall("poly$", v)]
        ["_".join([area, "poly"]) for area in highlight_areas]
        _ids = [_polys.index(i) for i in ["_".join([area, "poly"]) for area in highlight_areas]]
        _colors = np.array(["r"] * len(_polys))
        _colors[_ids] = "green"
        _alphas = np.array([0.05] * len(_polys))
        _alphas[_ids] = alpha
        for i, _poly in enumerate(_polys):
            ax.add_patch(PolygonPatch(eval(".".join(["self", _poly])), alpha=_alphas[i], color=_colors[i]))
        return ax, fig
    
    def get_area(self, regex: str = "crosswalk", tag_key: str = "name"):
        _ways, _nodes, _locs = [], [], []
        for (_, values) in self.osm_data["Relations"].items():
            tags = values["tags"]
            if tag_key not in tags.keys():
                break
            _found = re.findall(regex, tags[tag_key])
            if _found:
                _ways = [*_ways, *values["way_members"]]
        for _way in _ways:
            _nodes = [*_nodes, *self.osm_data["Ways"][_way]["nodes"]]
        for _node in _nodes:
            _locs.append(self.osm_data["Nodes"][_node])
        return _locs
    
    def __crosswalk_polygon(self):
        _points = self.get_area("crosswalk")
        crosswalk_shape = self.__get_exterior(_points)
        self.crosswalk_poly = crosswalk_shape.difference(self.intersection_poly) if re.findall("Tianjin", ROOT + self._map_dir) else crosswalk_shape

    def __intersection_polygon(self):
        _points = self.get_area("inter")
        self.intersection_poly = self.__get_exterior(_points)
    
    def __gap_polygon(self):
        _points = self.get_area("gap")
        self.gap_poly = self.__get_exterior(_points)
    
    def __road_and_sidewalk_polygon(self, sidewalk_size: float = 8.0):
        _points = self.get_area("")
        cross_poly = self.crosswalk_poly
        inter_poly = self.intersection_poly
        gap_poly = self.gap_poly
        road_poly = self.__get_exterior(_points)
        sidewalk_poly = road_poly.buffer(sidewalk_size).difference(road_poly).intersection(road_poly.minimum_rotated_rectangle.buffer(-0.2))
        self.road_poly, self.sidewalk_poly = road_poly.difference(cross_poly).difference(inter_poly).difference(gap_poly), sidewalk_poly

    def __fix_poly_sizes(self):
        self.intersection_poly = self.intersection_poly.difference(self.crosswalk_poly.buffer(3)).buffer(3)
        self.crosswalk_poly = self.crosswalk_poly.buffer(3).difference(self.intersection_poly).difference(self.road_poly).difference(self.sidewalk_poly).difference(self.gap_poly)

    def __initialize_polygons(self):
        self.__intersection_polygon(), self.__crosswalk_polygon(), self.__gap_polygon(), self.__road_and_sidewalk_polygon()
        self.__fix_poly_sizes()

    def __get_exterior(self, points: list, alpha: float = 0.4):
        return alphashape.alphashape(np.array(points), alpha=alpha)
    


class inD_map(SinD_map):
    def __init__(self):
        super().__init__(map_dir = "inD/lanelets/location2.osm")
    
    def roads(self):
        _nodes = self.get_area("road", "subtype")

    def walkway(self):
        _nodes = self.get_area("", "subtype")
        print(_nodes)
        


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
    map.plot_areas(highlight_areas=["road"])
    plt.show()
    