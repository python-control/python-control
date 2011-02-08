class Lti:
    """The Lti is a parent class to the StateSpace and TransferFunction child
    classes.  It only contains the number of inputs and outputs, but this can be
    expanded in the future."""
    
    def __init__(self, inputs=1, outputs=1):
        # Data members common to StateSpace and TransferFunction.
        self.inputs = inputs
        self.outputs = outputs
