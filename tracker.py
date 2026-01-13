import math

class Tracker:
    def __init__(self):
        self.center_points, self.id_count, self.disappeared, self.max_disappeared = {}, 0, {}, 10

    def register(self, cx, cy):
        self.center_points[self.id_count] = (cx, cy)
        self.disappeared[self.id_count] = 0
        self.id_count += 1

    def deregister(self, object_id):
        del self.center_points[object_id]; del self.disappeared[object_id]

    def update(self, objects_rect):
        objects_bbs_ids = []
        input_centroids = [((r[0]+r[2])//2, (r[1]+r[3])//2, r) for r in objects_rect]

        if not self.center_points:
            for cx, cy, r in input_centroids:
                self.register(cx, cy)
                objects_bbs_ids.append([*r, self.id_count - 1])
            return objects_bbs_ids

        object_ids = list(self.center_points.keys())
        if not input_centroids:
            for oid in object_ids:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared: self.deregister(oid)
            return objects_bbs_ids

        distances = []
        for i, (cx, cy, _) in enumerate(input_centroids):
            for oid in object_ids:
                d = math.hypot(cx - self.center_points[oid][0], cy - self.center_points[oid][1])
                if d < 80: distances.append((d, i, oid))

        distances.sort()
        used_rows, used_cols = set(), set()
        for d, r, c in distances:
            if r in used_rows or c in used_cols: continue
            self.center_points[c] = input_centroids[r][:2]
            self.disappeared[c] = 0
            objects_bbs_ids.append([*input_centroids[r][2], c])
            used_rows.add(r); used_cols.add(c)

        for r in range(len(input_centroids)):
            if r not in used_rows:
                self.register(*input_centroids[r][:2])
                objects_bbs_ids.append([*input_centroids[r][2], self.id_count - 1])

        for oid in object_ids:
            if oid not in used_cols:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared: self.deregister(oid)
        return objects_bbs_ids
