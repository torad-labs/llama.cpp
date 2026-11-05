// Goal: the operator can compare evidence, change assumptions, inspect events,
// and recover from invalid imports without leaking markup or losing the page.
const assert = require('node:assert/strict');
const { chromium } = require(require.resolve('playwright-core', {
  paths: [process.env.PLAYWRIGHT_NODE_PATH || process.cwd()]
}));
(async () => {
  const browser = await chromium.connectOverCDP(process.env.DECODE_LAB_CDP);
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.setViewportSize({ width: 3840, height: 2050 });
    await page.goto(process.env.DECODE_LAB_URL || 'http://127.0.0.1:8768');
    await page.waitForSelector('[data-ready="true"]', { timeout: 30000 });
    if (process.argv.includes('--future-only')) {
      assert.match(await page.locator('#ideal-workload').innerText(), /^40 active/);
      assert.match(await page.locator('#comparison-status').innerText(), /40-session throughput/);
      assert.equal(await page.locator('#ideal-gpus .gpu-chip').count(), 4);
      assert.equal(await page.locator('#ideal-gpus .unknown-chip').count(), 0);
      for (const gpu of await page.locator('#ideal-gpus .gpu-chip').allTextContents()) {
        assert.match(gpu, /10 owners/);
      }
      assert.equal(await page.locator('#profile-span').innerText(), '--');
      assert.match(await page.locator('#profile-workload').innerText(), /Expected: 40 active/);
      assert.equal(await page.locator('#profile-gpus [data-state="pending"]').count(), 4);
      assert.deepEqual(await page.locator('#future-cycle-path .phase-step b').allTextContents(),
        await page.locator('#cycle-path .phase-step b').allTextContents());
      assert.equal(await page.locator('#future-layer-map [data-state="pending"]').count(), 45);
      assert.equal(await page.locator('#visual-timeline').isVisible(), false);
      assert.equal(await page.locator('table:visible').count(), 0);
      await page.locator('#future-cycle-path button').first().click();
      assert.match(await page.locator('#visual-inspector').innerText(), /Awaiting capture/);
      assert.deepEqual(errors, []);
      console.log('PASS: ideal and empty future capture share GPU, phase and layer slots');
      return;
    }
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
      await page.waitForFunction(() => document.querySelector('#raw-status').textContent.includes('Batch settings match'));
      await page.locator('#raw-toggle').click();
      await page.locator('#view-kernels').click();
      await page.locator('#search').fill('probe');
      assert.match(await page.locator('#ledger').innerText(), /<img/);
      assert.equal(await page.evaluate(() => window.__decode_injected), undefined);
      assert.equal(await page.locator('#ledger img').count(), 0);
      assert.equal(await page.locator('#profile-span').innerText(), '--');
      assert.equal(await page.locator('#profile-gpus [data-state="pending"]').count(), 4);
      assert.deepEqual(errors, []);
      console.log('PASS: valid local import, matching batch status, untrusted labels rendered as text');
      return;
    }
    if (process.argv.includes('--phone-only')) {
      await page.setViewportSize({width:390,height:844});
      const recorded = await page.locator('#mobile-profile').boundingBox();
      assert.notEqual(recorded, null, 'Phone must show the recorded comparison at a glance');
      assert.ok(recorded.y + recorded.height <= 844, 'Both time shares fit in the first phone viewport');
      assert.deepEqual(errors, []);
      console.log('PASS: both workloads visible in the first phone viewport');
      return;
    }
    assert.match(await page.locator('#ideal-workload').innerText(), /400,000/);
    assert.match(await page.locator('#profile-workload').innerText(), /Expected: 40 active/);
    assert.equal(await page.locator('#visual-compare').isVisible(), true);
    assert.equal(await page.locator('table:visible').count(), 0);
    assert.equal(await page.locator('.gpu-chip').count(), 8);
    await page.locator('[data-inspect="transfer"]').first().click();
    assert.equal(await page.locator('#visual-inspector').isVisible(), true);
    assert.match(await page.locator('#visual-inspector').innerText(), /Not measured/);
    assert.ok((await page.locator('#visual-compare').boundingBox()).x + (await page.locator('#visual-compare').boundingBox()).width <= (await page.locator('#visual-inspector').boundingBox()).x,
      'Desk inspector must not cover the comparison');
    await page.locator('#visual-inspector-close').click();
    await page.locator('#ideal-gpus .gpu-chip').first().click();
    assert.match(await page.locator('#visual-inspector').innerText(), /100 squares|1%/);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#visual-inspector').isVisible(), false);
    assert.equal(await page.locator('#ideal-gpus .gpu-chip').first().evaluate(el => el === document.activeElement), true);
    await page.locator('#configure').click();
    await page.locator('#active').selectOption('1');
    await page.waitForFunction(() => document.querySelector('#ideal-workload').textContent.startsWith('1 active'));
    assert.equal(await page.locator('#ideal-gpus .unknown-chip').count(), 3);
    await page.locator('#active').selectOption('4');
    await page.waitForFunction(() => document.querySelector('#ideal-workload').textContent.startsWith('4 active'));
    await page.locator('#settings-close').click();
    await page.locator('#layer-map button').nth(3).click();
    await page.getByRole('button', {name:'Inspect every substep',exact:true}).click();
    assert.match(await page.locator('#ledger-title').innerText(), /Layer 3/);
    await page.locator('#raw-toggle').click();
    await page.locator('#raw-toggle').click();
    await page.locator('#capture').selectOption('timeline-4k');
    await page.waitForFunction(() => document.querySelector('#raw-workload').textContent.includes('4,096 context'));
    await page.locator('#visual-timeline .timeline-bin').first().focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#visual-timeline .timeline-bin').nth(1).evaluate(el => el === document.activeElement), true);
    await page.keyboard.press('Enter');
    assert.match(await page.locator('#visual-inspector').innerText(), /captured window/);
    await page.keyboard.press('Escape');
    await page.locator('#raw-data > details > summary').click();
    await page.locator('#activity-table summary').click();
    assert.equal(await page.locator('#activity-windows tbody tr').count(), 96);
    assert.match(await page.locator('#ledger').innerText(), /Ideal/);
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
    await page.waitForFunction(() => document.querySelector('#raw-workload').textContent.includes('512 context'));
    assert.equal(await page.locator('#profile-span').innerText(), '--');
    assert.equal(await page.locator('#profile-gpus [data-state="pending"]').count(), 4);
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
    await context.close();
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
