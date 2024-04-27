import general as GenFun

class SingleWire():
    """ A simple container for the catenary system containing all design data
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, pandasdesigndetails, dataframe):
        self._cp, self._acp, self._hd, self._bd, self._sl = create_df_dd(pandasdesigndetails)
        #self._cp = conductorparticulars
        #self._hd = sample_df_hd()
        #self._bd = sample_df_bd()
        self._wr = GenFun.WireRun_df(dataframe, self._bd.iloc[1,1])
        #[P_DiscreteLoad_CW, STA_DiscreteLoad_CW,
        # P_DiscreteLoad_MW, STA_DiscreteLoad_MW]
        self._solved = False
        self._catenarysag = None
        self._sag = None
        self._sag_w_ha = None
        self.empty = False

    def _solve(self):