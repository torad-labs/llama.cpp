// Goal: the operator can compare evidence, change assumptions, inspect events,
// and recover from invalid imports without leaking markup or losing the page.
const assert = require('node:assert/strict');
const { chromium } = require(require.resolve('playwright-core', {
  paths: [process.env.PLAYWRIGHT_NODE_PATH || process.cwd()]
}));
(async () => {
  const browser = await chromium.connectOverCDP(process.env.DECODE_LAB_CDP);
  const context = browser.contexts()[0];
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.setViewportSize({ width: 3840, height: 2050 });
    await page.goto(process.env.DECODE_LAB_URL || 'http://127.0.0.1:8768');
    await page.waitForSelector('[data-ready="true"]', { timeout: 30000 });
    if (process.argv.includes('--import-only')) {
      const fs = require('node:fs');
      const zlib = require('node:zlib');
      const path = require('node:path');
      const profile = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.resolve(__dirname,
        '../../docs/torad/decode-lab/data/timeline-4k.json.gz'))));
      profile.metadata.active = 40;
      profile.metadata.context = 400000;
      profile.metadata.depth_policy = 'fixed-2';
      profile.kernels[0].name = '<img src=x onerror="window.__decode_injected=1"> probe';
      await page.locator('#import-file').setInputFiles({
        name: 'new-profile.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(profile))
      });
      await page.waitForFunction(() => document.querySelector('#comparison-status').textContent.includes('Batch settings match'));
      await page.locator('#view-kernels').click();
      await page.locator('#search').fill('probe');
      assert.match(await page.locator('#ledger').innerText(), /<img/);
      assert.equal(await page.evaluate(() => window.__decode_injected), undefined);
      assert.equal(await page.locator('#ledger img').count(), 0);
      assert.deepEqual(errors, []);
      console.log('PASS: valid local import, matching batch status, untrusted labels rendered as text');
      return;
    }
    assert.match(await page.locator('#mismatch').innerText(), /400,000/);
    assert.match(await page.locator('#profile-workload').innerText(), /4,096/);
    assert.match(await page.locator('#ledger').innerText(), /Ideal/);
    await page.locator('#active').selectOption('4');
    await page.waitForFunction(() => document.querySelector('#ideal-workload').textContent.startsWith('4 active'));
    await page.locator('#view-kernels').click();
    await page.locator('#search').fill('ggml_cuda_ar');
    assert.ok(await page.locator('#ledger tbody tr').count() > 0);
    assert.match(await page.locator('#ledger').innerText(), /Not measured/);
    await page.locator('#ledger tbody button').first().click();
    assert.match(await page.locator('#inspector').innerText(), /Captured events/);
    await page.locator('#view-copies').click();
    await page.locator('#search').fill('');
    assert.match(await page.locator('#ledger').innerText(), /GB\/s/);
    await page.locator('#capture').selectOption('timeline-512');
    await page.waitForFunction(() => document.querySelector('#profile-workload').textContent.includes('512 context'));
    await page.locator('#import-file').setInputFiles({
      name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"schema":"wrong"}')
    });
    await page.waitForSelector('#load-error:not([hidden])');
    assert.match(await page.locator('#load-error').innerText(), /profile|export/i);
    assert.ok(await page.locator('#ledger tbody tr').count() > 0);
    for (const width of [3840, 1440, 390]) {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true,
        'Page overflows at ' + width);
    }
    assert.deepEqual(errors, []);
    console.log('PASS: comparison, assumptions, kernels, raw events, copies, capture switch, invalid import, responsive layout');
  } finally {
    await page.close();
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
