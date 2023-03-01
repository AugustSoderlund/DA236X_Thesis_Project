import time
import osmium as osm
import pandas as pd

ROOT = "C:/Users/ASOWXU/Desktop/Thesis Project/Code/DA236X_Thesis_Project/thesis_project/.datasets/SinD/Data/mapfile-Tianjin.osm"

class SinD_map:
    def __init__(self, map_dir: str = ROOT):
        self.tag_df = self.__load_tags_to_df(map_dir)

    def __load_tags_to_df(self, map_dir: str):
        osmhandler = OSMHandler()

        # scan the input file and fills the handler list accordingly
        osmhandler.apply_file(map_dir)
        print(osmhandler.nodes)

        # transform the list into a pandas DataFrame
        data_colnames = ['type', 'id', '__', 'visible', '__', '__',
                        '__', '__', '__', 'tagkey', 'tagvalue']
        df_osm = pd.DataFrame(osmhandler.osm_data, columns=data_colnames)
        return df_osm


class OSMHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.osm_data = []
        self.nodes = {}
        self.ways = {}
        self.relations = {}

    def tag_inventory(self, elem, elem_type):
        if type(elem) == osm.osmium.osm.types.Node:
            self.nodes.update({elem.id: [elem.location.x, elem.location.y]})
        for tag in elem.tags:
            self.osm_data.append([elem_type, 
                                   elem.id, 
                                   elem.version,
                                   elem.visible,
                                   pd.Timestamp(elem.timestamp),
                                   elem.uid,
                                   elem.user,
                                   elem.changeset,
                                   len(elem.tags),
                                   tag.k, 
                                   tag.v])

    def node(self, n):
        self.tag_inventory(n, "node")

    def way(self, w):
        self.tag_inventory(w, "way")

    def relation(self, r):
        self.tag_inventory(r, "relation")

if __name__ == "__main__":
    map = SinD_map()
    print(map.tag_df)