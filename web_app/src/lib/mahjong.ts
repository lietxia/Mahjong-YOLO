import { Calculator } from '../../vendor/mahjong-vue/src/store/calc';
import { Rule, type Result } from '../../vendor/mahjong-vue/src/store/definition';
import {
  Pai,
  PositionType,
  RON,
  TSUMO,
  IPPATSU,
  RIICHI,
  State,
  type PaiType,
} from '../../vendor/mahjong-vue/src/store/types';
import { countRedDora, hasUnsupportedTiles } from './tile';

export type Wind = 'east' | 'south' | 'west' | 'north';
export type AgariType = 'tsumo' | 'ron';

export type ScoreContext = {
  fieldWind: Wind;
  seatWind: Wind;
  agariType: AgariType;
  riichi: boolean;
  ippatsu: boolean;
  agariIndex: number;
  doraIndicators: string[];
  uraIndicators: string[];
};

export type ScoreComputation = {
  status: 'ready' | 'incomplete';
  message: string;
  warnings: string[];
  normalizedTiles: string[];
  redDoraCount: number;
  result?: Result;
};

function mapWindToPosition(wind: Wind): PositionType {
  switch (wind) {
    case 'east':
      return PositionType.EAST;
    case 'south':
      return PositionType.SOUTH;
    case 'west':
      return PositionType.WEST;
    case 'north':
      return PositionType.NORTH;
  }
}

function normalizeTileCode(tile: string): { pai: Pai; isRed: boolean } {
  const type = tile.slice(-1) as PaiType;
  const rawNumber = Number(tile[0]);
  const isRed = rawNumber === 0;
  const pai = new Pai(type, isRed ? 5 : rawNumber);
  if (isRed) {
    pai.isRed = true;
    pai.redCnt = 1;
  }
  return { pai, isRed };
}

function normalizeIndicatorList(values: string[]): Pai[] {
  return values
    .map((value) => value.trim())
    .filter(Boolean)
    .filter((value) => /^(?:[1-9]|0)[mpsz]$/.test(value))
    .map((value) => normalizeTileCode(value).pai);
}

function buildYakuFlags(context: ScoreContext): number[] {
  const flags: number[] = [];

  if (context.riichi) {
    flags.push(RIICHI);
  }

  if (context.ippatsu) {
    flags.push(IPPATSU);
  }

  return flags;
}

export function calculateMahjongScore(tileLabels: string[], context: ScoreContext): ScoreComputation {
  const warnings: string[] = [];
  const normalizedTiles = tileLabels.filter(Boolean);

  if (normalizedTiles.length !== 14) {
    return {
      status: 'incomplete',
      message: `当前识别到 ${normalizedTiles.length} 张牌。Phase 1 只支持 14 张闭门和牌计算。`,
      warnings,
      normalizedTiles,
      redDoraCount: countRedDora(normalizedTiles),
    };
  }

  if (hasUnsupportedTiles(normalizedTiles)) {
    return {
      status: 'incomplete',
      message: '识别结果里存在无法映射到麻将牌编码的标签，暂时不能进入和牌计算。',
      warnings,
      normalizedTiles,
      redDoraCount: countRedDora(normalizedTiles),
    };
  }

  if (context.agariIndex < 0 || context.agariIndex >= normalizedTiles.length) {
    return {
      status: 'incomplete',
      message: '和牌索引越界，请重新选择和牌牌张。',
      warnings,
      normalizedTiles,
      redDoraCount: countRedDora(normalizedTiles),
    };
  }

  const agariCode = normalizedTiles[context.agariIndex];
  const concealedCodes = normalizedTiles.filter((_, index) => index !== context.agariIndex);
  const concealedPais = concealedCodes.map((code) => normalizeTileCode(code).pai);
  const agariPai = normalizeTileCode(agariCode).pai;
  const redDoraCount = countRedDora(normalizedTiles);

  const state = new State(
    mapWindToPosition(context.fieldWind),
    mapWindToPosition(context.seatWind),
    buildYakuFlags(context),
    context.agariType === 'tsumo' ? TSUMO : RON,
    concealedPais,
    [],
    normalizeIndicatorList(context.doraIndicators),
    normalizeIndicatorList(context.uraIndicators),
    agariPai,
    redDoraCount,
  );

  const calculator = new Calculator();
  const result = calculator.calculate(state, new Rule());

  if (result.han === 0 && result.fu === 0 && result.yaku.length === 0) {
    warnings.push('当前牌型没有被判定为可和牌型或无役。');
  }

  return {
    status: 'ready',
    message: warnings.length > 0 ? '已完成计算，但结果需要结合提示一起判断。' : '已完成 Phase 1 和牌计算。',
    warnings,
    normalizedTiles,
    redDoraCount,
    result,
  };
}
