import heapq

class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
        self.index = None
        self.vector = None

    def __lt__(self, other):
        if other is None:
            return -1
        if not isinstance(other, HeapNode):
            return -1
        return self.freq < other.freq


class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        self.merged_nodes = None

    def make_heap(self, frequency):  # frequency has a shape of  { word : frequency }
        for key in frequency:  # make a node, then push in list of heap queue
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):      # make nodes from low to high frequency and merge to tree.
        index = 0
        merged = None
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            merged.index = index                # index is reversed, i.e. root node has a biggest index.
            heapq.heappush(self.heap, merged)

            index += 1

        return merged

    def make_codes_helper(self, root, current_code):
        if root is None:
            return

        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def build(self, frequency):
        self.make_heap(frequency)  # 최소 힙 트리 생성
        merged = self.merge_nodes()  # 작은것부터 두개씩 pop 하고 key=None 인 새로운 노드 할당 후 자식으로 넣음.
        self.make_codes()  # 만들어진 이진트리에서 root~each word 까지 가는 길 coding (왼쪽은 0, 오른쪽은 1)

        return self.codes, merged
