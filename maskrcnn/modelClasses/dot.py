import drawSvg as draw


class Dot(draw.DrawingParentElement):
    TAG_NAME = 'a'

    def __init__(self, id, **kwargs):
        # Other init logic...
        # Keyword arguments to super().__init__() correspond to SVG node
        # arguments: stroke_width=5 -> stroke-width="5"
        super().__init__(id=id, **kwargs)
