from pade.core.agent import Agent
from pade.acl.aid import AID


class Mas:

    def __init__(self, agents, links, method):
        self.agents = agents
        self.links = links
        self.method = method
