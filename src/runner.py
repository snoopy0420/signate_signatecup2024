import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import sys,os
import shap
import yaml
from tqdm import tqdm, tqdm_notebook
# from sklearn.metrics import mean_absolute_error
from typing import Callable, List, Optional, Tuple, Union
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
FIGURE_DIR_NAME = yml['SETTING']['DIR_FIGURE']
DIR_HOME = yml['SETTING']['DIR_HOME']
DIR_MODEL = yml['SETTING']['DIR_MODEL']
DIR_FIGURE = yml['SETTING']['DIR_FIGURE']

sys.path.append(DIR_HOME)
from src.model import Model
from src.util import Util, Metric, Validation


# 定数
shap_sampling = 10000

class Runner:
    """学習・予測・評価・パラメータチューニングを担うクラス
    """

    # コンストラクタ
    def __init__(self,
                 run_name: str, # runの名前
                 model_cls: Callable[[str, dict], Model], #モデルのクラス
                 params: dict, # ハイパーパラメータ
                 df_train: pd.DataFrame, # 学習データ
                 df_test: pd.DataFrame, # テストデータ
                 run_setting: dict,
                 logger,
                 ): 
        
        # 評価関数
        self.metrics = roc_auc_score
        self.run_name = run_name
        self.model_cls = model_cls
        self.params = params
        self.key = run_setting.get('key')
        self.target = run_setting.get('target')
        self.calc_shap = run_setting.get('calc_shap')
        self.save_train_pred = run_setting.get('save_train_pred')
        self.hopt = run_setting.get("hopt")
        self.target_enc = run_setting.get("target_enc")
        self.target_enc_col = run_setting.get("target_enc_col")
        # CVの設定
        self.n_splits = 5
        self.random_state = 2024
        self.shuffle = True
        # データのセット
        self.train_x = df_train.drop(columns=[self.target])
        self.test_x = df_test.drop(columns=[self.target])
        self.train_y = df_train[self.target]
        self.test_y = df_test[self.target]
        self.out_dir_name = DIR_MODEL
        self.logger = logger
        if self.calc_shap:
            self.shap_values = np.zeros(self.train_x.shape)
        if self.hopt != False:
            # self.paramsを上書きする
            self.run_hopt()
            self.shap_values = np.zeros(self.train_x.shape)

    
    def train_fold(self, i_fold: Union[int, str], metrics=None) -> Tuple[Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """foldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）, metrics: 評価に用いる関数
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """

        # データセットの準備
        # 学習データ・バリデーションデータのindexを取得
        tr_idx, va_idx = Validation.load_index_k_fold(i_fold, self.train_x, self.n_splits, self.shuffle, self.random_state)
        # 学習データ・バリデーションデータをセットする
        tr_x, tr_y = self.train_x.iloc[tr_idx], self.train_y.iloc[tr_idx]
        va_x, va_y = self.train_x.iloc[va_idx], self.train_y.iloc[va_idx]

        # target encoding
        if self.target_enc:
            tr_x, va_x = self.get_target_encoding(tr_x, tr_y, va_x, self.target_enc_col) 

        # 学習を行う
        model = self.build_model(i_fold)
        model.train(tr_x, tr_y, va_x, va_y)

        # バリデーションデータの予測
        if self.calc_shap:
            va_pred, self.shap_values[va_idx[:shap_sampling]] = model.predict_and_shap(va_x, shap_sampling)
        else:
            va_pred = model.predict(va_x)

        # バリデーションデータの評価
        score = self.metrics(va_y, va_pred)

        # モデル、インデックス、予測値、評価を返す
        return model, va_idx, va_pred, score

    
    def run_train_cv(self) -> None:
        """CVでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        # ログ
        self.logger.info(f'{self.run_name} - start training cv')

        scores = [] # 各foldのscoreを保存
        va_idxes = [] # 各foldのvalidationデータのindexを保存
        va_preds = [] # 各foldのバリデーションデータに対する予測値を保存

        # fold毎の学習：train_foldをn_splits回繰り返す
        for i_fold in range(self.n_splits):

            # 学習を行う
            self.logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            self.logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            scores.append(score)
            va_idxes.append(va_idx)
            va_preds.append(va_pred)

        self.logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}') # スコアを出力

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        va_preds = np.concatenate(va_preds, axis=0)
        va_preds = va_preds[order]

        # 学習データでの予測結果の保存
        if self.save_train_pred:
            Util.dump_df_pickle(pd.DataFrame(va_preds), self.out_dir_name + f'{self.run_name}-train_preds.pkl')

        # 評価結果の保存
        self.logger.result_scores(self.run_name, scores)
        self.logger.result("mean: {}, std: {}".format(np.mean(scores), np.std(scores)))

        # shap feature importanceデータの保存
        if self.calc_shap:
            self.shap_feature_importance() # shap値の計算・可視化


    def run_predict_cv(self) -> None:
        """CVで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        self.logger.info(f'{self.run_name} - start prediction cv')
        preds = []

        # target encoding
        if self.target_enc:
            self.test_x = self.get_test_target_enc(self.train_x, self.train_y,self.test_x, self.target_enc_col) 

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_splits):
            self.logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(self.test_x)
            preds.append(pred)
            self.logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        df_pred = pd.concat([self.test_x[self.key].reset_index(drop=True),
                             pd.DataFrame({"pred":pred_avg}).reset_index(drop=True)], axis=1)

        # 予測結果の保存
        Util.dump_df_pickle(df_pred, os.path.join(self.out_dir_name, f'{self.run_name}-pred.pkl'))

        self.logger.info(f'{self.run_name} - end prediction cv')


    def plot_feature_importance_cv(self) -> None:
        """CVで学習した各foldのモデルの平均により、特徴量の重要度を取得する
        """
        list_feat_imp = []
        for i_fold in range(self.n_splits):
            model = self.build_model(i_fold)
            model.load_model()
            list_feat_imp.append(model.get_feature_importance())
        df_feat_imp = pd.concat([pd.Series(feat_imp) for feat_imp in list_feat_imp], axis=1)

        # 各foldの平均を算出
        # 各foldの標準偏差を算出
        df_feat_imp_ = pd.DataFrame({
            'feature': self.train_x.columns,
            'mean': df_feat_imp.mean(axis=1),
            'std': df_feat_imp.std(axis=1)
        }).sort_values('mean')

        df = df_feat_imp_
        # 変動係数を算出
        df['coef_of_var'] = df['std'] / df['mean']
        df['coef_of_var'] = df['coef_of_var'].fillna(0)
        df = df.sort_values('mean', ascending=True)

        # 出力
        fig, ax1 = plt.subplots(figsize = (10, 30))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()

        # 棒グラフを出力
        ax1.set_title('feature importance gain')
        ax1.set_xlabel('feature importance mean & std')
        ax1.barh(df["feature"], df['mean'], label='mean',  align="center", alpha=0.6)
        ax1.barh(df["feature"], df['std'], label='std',  align="center", alpha=0.6)

        # 折れ線グラフを出力
        ax2 = ax1.twiny()
        ax2.plot(df['coef_of_var'], df["feature"], linewidth=1, color="crimson", marker="o", markersize=8, label='coef_of_var')
        ax2.set_xlabel('Coefficient of variation')

        # 凡例を表示（グラフ左上、ax2をax1のやや下に持っていく）
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=12)
        ax2.legend(bbox_to_anchor=(1, 0.97), loc='upper right', borderaxespad=0.5, fontsize=12)

        # グリッド表示(ax1のみ)
        ax1.grid(True)
        ax2.grid(False)

        # 図を保存
        plt.savefig(os.path.join(DIR_FIGURE, f'{self.run_name}_fi_gain.png'), dpi=300, bbox_inches="tight")
        plt.close()




####### model utils ##################################################################

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # run名、i_fold、モデルのクラス名からモデルを作成する
        run_fold_name = f'{self.run_name}-fold{i_fold}'
        model = self.model_cls(run_fold_name, self.params)
        return model


    def shap_feature_importance(self) -> None:
        """計算したshap値を可視化して保存する
        """
        all_columns = self.train_x.columns.values.tolist() + [self.target]
        ma_shap = pd.DataFrame(sorted(zip(abs(self.shap_values).mean(axis=0), all_columns), reverse=True),
                        columns=['Mean Abs Shapley', 'Feature']).set_index('Feature')
        ma_shap = ma_shap.sort_values('Mean Abs Shapley', ascending=True)

        # 可視化
        fig = plt.figure(figsize = (8,30))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        ax = fig.add_subplot(1,1,1)
        ax.set_title('shap value')
        ax.barh(ma_shap.index, ma_shap['Mean Abs Shapley'] , label='Mean Abs Shapley',  align="center", alpha=0.8)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=0, fontsize=10)
        ax.legend(loc = 'upper left')
        plt.savefig(FIGURE_DIR_NAME + self.run_name + '_shap.png', dpi=300, bbox_inches="tight")
        plt.close()


    def run_hopt(self):
        """パラメータチューニングを行い、self.paramsを上書きする
        """
        # モデルにパラメータを指定して学習・予測させた場合のスコアを返す関数を定義
        def score(params):
            self.logger.info(f'{self.run_name} - start hopt eval - params  {params}')

            if (self.hopt == "xgb_hopt") | (self.hopt == "lgb_hopt"):
                params["max_depth"] = int(params["max_depth"])
            if (self.hopt == "lgb_hopt"):
                params["num_leaves"] = int(params["num_leaves"])
                params["min_data_in_leaf"] = int(params["min_data_in_leaf"])

            for param in params.keys():
                self.params[param] = params[param] # 実験するパラメータをセット
            i_fold = np.random.randint(0, self.n_splits) # foldの選択はランダムにする
            model, va_idx, va_pred, score = self.train_fold(i_fold, metrics=self.metrics) # 1foldで学習
            self.logger.info(f'{self.run_name} - end hopt eval - score {score}')

            # 情報を記録しておく
            history.append((params, score))

            return {'loss': score, 'status': STATUS_OK}

        # 探索空間の設定
        if self.hopt == "xgb_hopt":
            param_space = {
                'min_child_weight': hp.loguniform('min_child_weight', np.log(0.1), np.log(10)),
                'max_depth': hp.quniform('max_depth', 3, 9, 1),
                'subsample': hp.quniform('subsample', 0.6, 0.95, 0.05),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.95, 0.05),
                'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
                # 余裕があればalpha, lambdaも調整する
                'alpha' : hp.loguniform('alpha', np.log(1e-8), np.log(1.0)),
                'lambda' : hp.loguniform('lambda', np.log(1e-6), np.log(10.0)),
            }
        elif self.hopt == "lgb_hopt":
            param_space = {
                'num_leaves': hp.quniform('num_leaves', 50, 200, 10),
                'max_depth': hp.quniform('max_depth', 3, 10, 1),
                'min_data_in_leaf': hp.quniform('min_data_in_leaf',  5, 25, 2),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
                'subsample': hp.uniform('subsample', 0.5, 1.0)  
            }
        elif self.hopt == "nn_hopt":
            param_space = {
                'input_dropout': hp.quniform('input_dropout', 0, 0.2, 0.05),
                'hidden_layers': hp.quniform('hidden_layers', 2, 4, 1),
                'hidden_units': hp.quniform('hidden_units', 32, 256, 32),
                'hidden_activation': hp.choice('hidden_activation', ['prelu', 'relu']),
                'hidden_dropout': hp.quniform('hidden_dropout', 0, 0.3, 0.05),
                'batch_norm': hp.choice('batch_norm', ['before_act', 'no']),
                'optimizer': hp.choice('optimizer', [{'type': 'adam', 'lr': hp.loguniform('adam_lr', np.log(0.000001), np.log(0.001))},
                                                     {'type': 'sgd', 'lr': hp.loguniform('sgd_lr', np.log(0.000001), np.log(0.001))}]),
                'batch_size': hp.quniform('batch_size', 32, 128, 32),
            }


        # hyperoptによるパラメータ探索の実行
        max_evals = 25  # 試行回数
        trials = Trials()
        history = []
        np.random.seed(20) # foldの選択seed
        fmin(score, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

        # 記録した情報からパラメータとスコアを出力する
        history = sorted(history, key=lambda tpl: tpl[1])
        best = history[0]
        self.logger.info(f'{self.run_name} - best paramns {best[0]}')
        self.logger.info(f'{self.run_name} - best score {best[1]}')

        # self.paramsを上書き
        for param in best[0].keys():
            self.params[param] = best[0][param]

    
    def get_target_encoding(self, tr_x, tr_y, va_x, cat_cols):
        """学習データとバリデーションデータのtarget encodingを実行する
        """
        if cat_cols == "all":
            cat_cols = list(tr_x.dtypes[tr_x.dtypes=="object"].index) # 全てのカテゴリ
            # cat_cols = list(tr_x.columns) # 全てのカラム

        # 変数をループしてtarget encoding
        print("target_encoding", cat_cols)
        for c in cat_cols:
            data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})

            # バリデーションデータ(va_x)を変換
            # 学習データ全体で各カテゴリにおけるtargetの平均を計算
            target_mean = data_tmp.groupby(c)['target'].mean()
            va_x.loc[:, c] = va_x[c].map(target_mean)  # 置換
            # va_x = va_x.fillna(data_tmp["target"].mean()) # nanになってしまったところは平均値で埋める

            # 学習データ(tr_x)を変換
            tmp = np.repeat(np.nan, tr_x.shape[0]) # 変換後の値を格納する配列を準備
            kf_encoding = KFold(n_splits=10, shuffle=True, random_state=72)
            for idx_1, idx_2 in kf_encoding.split(tr_x):
                # out-of-foldで各カテゴリにおける目的変数の平均を計算
                target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
                # 変換後の値を一時配列に格納
                tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)
            tr_x.loc[:, c] = tmp# 置換
            # tr_x = tr_x.fillna(data_tmp["target"].mean()) # nanになってしまったところは平均値で埋める

        return tr_x, va_x

    def get_test_target_enc(self, train_x, train_y, test_x, cat_cols):
        """テストデータのtarget encodingを実行する
        """
        if cat_cols == "all":
            cat_cols = list(train_x.dtypes[train_x.dtypes=="object"].index)
            # cat_cols = list(train_x.columns) # 全てのカラム

        for c in cat_cols:
            # テストデータを変換
            data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
            target_mean = data_tmp.groupby(c)['target'].mean()
            test_x[c] = test_x[c].map(target_mean)
            test_x = test_x.fillna(data_tmp["target"].mean()) # nanになってしまったところは平均値で埋める

        return test_x


    def get_feature_name(self):
        """ 学習に使用した特徴量を返却
        """
        return self.train_x.columns.values.tolist()

    def get_params(self):
        """ 学習に使用したハイパーパラメータを返却
        """
        return self.params
