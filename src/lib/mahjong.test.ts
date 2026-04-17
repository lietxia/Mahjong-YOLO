import { describe, expect, it } from 'vitest';
import { calculateMahjongScore, type FuroMeld } from './mahjong';

describe('mahjong score adapter', () => {
  it('returns incomplete when tile count is not 14', () => {
    const result = calculateMahjongScore(['1m', '2m'], {
      fieldWind: 'east',
      seatWind: 'east',
      agariType: 'tsumo',
      riichi: false,
      ippatsu: false,
      agariIndex: 1,
      doraIndicators: [],
      uraIndicators: [],
      furo: [],
    });

    expect(result.status).toBe('incomplete');
  });

  it('returns ready for a 14-tile hand input', () => {
    const result = calculateMahjongScore(
      ['1m', '2m', '3m', '4m', '5m', '6m', '2p', '3p', '4p', '7s', '8s', '9s', '5z', '5z'],
      {
        fieldWind: 'east',
        seatWind: 'east',
        agariType: 'ron',
        riichi: false,
        ippatsu: false,
        agariIndex: 13,
        doraIndicators: [],
        uraIndicators: [],
        furo: [],
      },
    );

    expect(result.status).toBe('ready');
    expect(result.normalizedTiles).toHaveLength(14);
    expect(result.result).toBeDefined();
  });

  it('returns ready for a hand with pon furo (11 concealed + 1 pon)', () => {
    const result = calculateMahjongScore(
      ['1m', '2m', '3m', '4m', '5m', '6m', '7s', '8s', '9s', '5z', '5z'],
      {
        fieldWind: 'east',
        seatWind: 'east',
        agariType: 'ron',
        riichi: false,
        ippatsu: false,
        agariIndex: 10,
        doraIndicators: [],
        uraIndicators: [],
        furo: [{ type: 'pon', tiles: ['2p', '2p', '2p'] }],
      },
    );

    expect(result.status).toBe('ready');
    expect(result.normalizedTiles).toHaveLength(11);
    expect(result.result).toBeDefined();
  });

  it('returns ready for a hand with chi furo', () => {
    const result = calculateMahjongScore(
      ['1m', '2m', '3m', '4m', '5m', '6m', '7s', '8s', '9s', '5z', '5z'],
      {
        fieldWind: 'east',
        seatWind: 'east',
        agariType: 'ron',
        riichi: false,
        ippatsu: false,
        agariIndex: 10,
        doraIndicators: [],
        uraIndicators: [],
        furo: [{ type: 'chi', tiles: ['2p', '3p', '4p'] }],
      },
    );

    expect(result.status).toBe('ready');
    expect(result.result).toBeDefined();
  });

  it('returns ready for a hand with kan furo (11 concealed + 1 kan = 15 total)', () => {
    const result = calculateMahjongScore(
      ['1m', '2m', '3m', '4m', '5m', '6m', '7s', '8s', '9s', '5z', '5z'],
      {
        fieldWind: 'east',
        seatWind: 'east',
        agariType: 'ron',
        riichi: false,
        ippatsu: false,
        agariIndex: 10,
        doraIndicators: [],
        uraIndicators: [],
        furo: [{ type: 'kan', tiles: ['2p', '2p', '2p', '2p'] }],
      },
    );

    expect(result.status).toBe('ready');
    expect(result.normalizedTiles).toHaveLength(11);
    expect(result.result).toBeDefined();
  });

  it('returns incomplete when concealed + furo does not add up', () => {
    const result = calculateMahjongScore(
      ['1m', '2m', '3m'],
      {
        fieldWind: 'east',
        seatWind: 'east',
        agariType: 'ron',
        riichi: false,
        ippatsu: false,
        agariIndex: 2,
        doraIndicators: [],
        uraIndicators: [],
        furo: [{ type: 'pon', tiles: ['2p', '2p', '2p'] }],
      },
    );

    expect(result.status).toBe('incomplete');
  });
});
