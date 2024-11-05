from netgen.thirdparty.chemprop_ng import chemprop
import os
from typing import List

class MLPredictor:
    """
    A ml predictor
    """
    def __init__(self, cp_path: str = None):
        if cp_path is None:
            curr_path = os.path.split(os.path.abspath(__file__))[0]
            net_gen_path = os.path.dirname(curr_path)
            cp_path = os.path.join(net_gen_path, 'thirdparty/chemprop_ng/trained_models/bimol01')

        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--features_path', '/dev/null',
            '--checkpoint_dir', cp_path,
            '--uncertainty_method', 'evidential_total',
        ]
        self.args = chemprop.args.PredictArgs().parse_args(arguments)
        # self.model_objects = chemprop.train.load_model(args=self.args)

    def predict(self, smiles: List[List[str]], temperature: float = None):

        if temperature is not None:
            features = [[temperature / 1000] for _ in range(len(smiles))]
        else:
            features = None

        preds = chemprop.train.make_predictions(args=self.args,
                                                smiles=smiles,
                                                return_uncertainty=True,
                                                features=features)

        return preds


if __name__ == '__main__':
    smiles = [['[H:1].[O:2]>>[H:1][O:2]'],
              ['[C:1]([H:2])[H:3].[H:4][H:5]>>[C:1]([H:2])([H:3])[H:4].[H:5]'],
              ['[H:1].[H:2][O:3][O:4]>>[H:1][H:2].[O:3][O:4]']]
    temperature = 850
    predictor = MLPredictor()
    preds, uncs = predictor.predict(smiles=smiles, temperature=temperature)

    print(preds, uncs)
"""
([[7.939595604951797], [13.455832409284753], [16.73729899560971]],
 [[6.300742417957018], [1.0463336762172162], [1.630387484295568]])
"""