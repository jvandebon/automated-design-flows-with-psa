from flask import Flask, render_template, request
import threading
import sys, os.path as osp
import inspect
from .hgraph import HGraph
from .utils.notebook import is_notebook

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class WebView:

    obj = None

    @staticmethod
    def req_index():
        return render_template('view.html')


    @staticmethod
    def req_hover(elem):

        if ',' in elem:
            elem = eval('(' + elem + ')')
        else:
            elem = int(elem)

        g = WebView.obj.graphs[WebView.obj.index]

        on_hover = WebView.obj.on_hovers[WebView.obj.index]

        # prevent from invoking multiple times, if the item is different
        if on_hover is not None:
            if (WebView.obj.hover_elem is None) or (elem != WebView.obj.hover_elem):
                if on_hover(g, elem):
                    WebView.obj.hover_elem = elem

        return { }



    @staticmethod
    def req_first():
        WebView.obj._reset()
        return WebView.obj.req_next()


    @staticmethod
    def payload(g):
        index = WebView.obj.index
        pl = {'svg': g,
              'index': index,
              'title': WebView.obj.titles[index],
              'ngraphs': len(WebView.obj.svg_graphs),
              'hover': WebView.obj.on_hovers[index] is not None
              }
        return pl

    @staticmethod
    def req_next():
        g = WebView.obj._next_graph()
        return WebView.payload(g)

    @staticmethod
    def req_prev():
        g = WebView.obj._prev_graph()
        return WebView.payload(g)

    @staticmethod
    def req_shutdown():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        return "Server shutting down..."


    def __init__(self, root_path=None):
        if root_path is None:
           root_path = osp.join(osp.abspath(osp.dirname(__file__)), 'assets')

        self.svg_graphs = []
        self.titles = []
        self.graphs = []
        self.on_hovers = [] # each graph has its hover
        self.hover_elem = None
        self.web_app = Flask('WebView', root_path=root_path)
        self.web_app.add_url_rule('/', 'index', self.__class__.req_index)
        self.web_app.add_url_rule('/first/', 'first', self.__class__.req_first)
        self.web_app.add_url_rule('/first', 'first', self.__class__.req_first)
        self.web_app.add_url_rule('/next/', 'next', self.__class__.req_next)
        self.web_app.add_url_rule('/next', 'next', self.__class__.req_next)
        self.web_app.add_url_rule('/prev', 'prev', self.__class__.req_prev)
        self.web_app.add_url_rule('/prev/', 'prev', self.__class__.req_prev)
        self.web_app.add_url_rule('/shutdown', 'shutdown', self.__class__.req_shutdown, methods=['POST'])
        self.web_app.add_url_rule('/shutdown/', 'shutdown', self.__class__.req_shutdown, methods=['POST'])
        self.web_app.add_url_rule('/hover/<elem>', 'hover', self.__class__.req_hover, methods=['GET'])

        WebView.obj = self

        self.index = -1

    def _reset(self):
        self.index = -1

    @property
    def graph(self):
        if self.index == -1:
            raise RuntimeError("invalid graph access!")
        return self.graphs[self.index]

    def _prev_graph(self):
        ngraphs = len(self.svg_graphs)
        if ngraphs == 0:
            self.index = -1
            return None

        self.index = (self.index - 1) % ngraphs
        return self.svg_graphs[self.index]

    def _next_graph(self):
        ngraphs = len(self.svg_graphs)
        if ngraphs == 0:
            self.index = -1
            return None

        self.index = (self.index + 1) % ngraphs
        return self.svg_graphs[self.index]

    def add_graph(self, graph, title='', **kwargs):
        #g=graph.copy()
        self.graphs.append(graph)
        self.titles.append(title)
        if '!on_hover' in graph.style:
            self.on_hovers.append(graph.style['!on_hover'])
        else:
            self.on_hovers.append(None)

        svg = graph.render(format="svg", pipe=True, **kwargs)
        self.svg_graphs.append(svg.decode('utf8'))


    def run(self, host='0.0.0.0', port='8888'):
        if is_notebook():
            import IPython.display 
            IPython.display.display(IPython.display.SVG(*self.svg_graphs))
            return

        host = '0.0.0.0' if host == None else host
        port = '8888' if port == None else port
        self._next_graph()
        print(f"URL: =====[http://{host}:{port}]=====")
        x = threading.Thread(target=self.web_app.run, kwargs=dict(host=host, port=port))
        x.start()
        x.join()
        #self.web_app.run(host=self.host, port=self.port, use_reloader=True)








