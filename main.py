import HeatDiffusion
import WaveEquation
import ReactionDiffusion
    
if __name__ == "__main__":

    Model = HeatDiffusion.HeatDiffusion()
    #Model = WaveEquation.WaveEquation()
    #Model = ReactionDiffusion.ReactionDiffusion()

    Model.Initialize()

    Mode = "Display"
    #Mode = "MakeGif"
    Model.Output(Mode)
