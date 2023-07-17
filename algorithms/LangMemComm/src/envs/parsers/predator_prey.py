


class PredatorPrey_Parser():

    def __init__(self, grid_shape):
        self.grid_shape = grid_shape

    def parse_global_state(self, state):
        """
        Parse the global state of the environment to produce a complete textual
        description of it.
        :param state: (np.ndarray) 2D map of the environment.
        :return descr: (list(list(str))) List of sentences describing the
            environment.
        """

        # Est-ce qu'on a vraiment besoin de la description globale ?
        # Commencer par le parser d'observations locales
        