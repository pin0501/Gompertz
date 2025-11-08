// Sample data
const SAMPLE_DATA = {
  periods: ['Q1 2020', 'Q2 2020', 'Q3 2020', 'Q4 2020', 'Q1 2021', 'Q2 2021', 'Q3 2021', 'Q4 2021', 'Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022', 'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
  accounts: [28988, 75699, 170312, 323900, 475110, 684196, 992318, 1191417, 1331101, 1601099, 1690404, 1833495, 2024223, 1900206, 1995234, 2186658, 2186357, 2382030, 2270589, 2235095]
};

// Application state
const state = {
  currentStep: 1,
  historicalData: [],
  dataQuality: null,
  fittedParams: null,
  forecastData: null,
  scenarios: ['conservative', 'moderate', 'aggressive'],
  scenarioParams: {
    conservative: { alpha: 0.10, delta_t: 0.5, kappa: 0.02, half_life: 6, window_length: 6, expansion_length: 8 },
    moderate: { alpha: 0.20, delta_t: 1.0, kappa: 0.05, half_life: 8, window_length: 8, expansion_length: 10 },
    aggressive: { alpha: 0.35, delta_t: 1.5, kappa: 0.08, half_life: 10, window_length: 12, expansion_length: 12 }
  },
  platformLaunchQuarter: 1,
  interpolationMethod: 'ignore'
};

let fittingChart = null;
let forecastChart = null;
let incrementalChart = null;

// Utility functions
function formatNumber(num) {
  if (!isFinite(num)) {
    return 'N/A';
  }
  if (num < 0) {
    num = 0;
  }
  return new Intl.NumberFormat('zh-TW').format(Math.round(num));
}

function parseQuarter(quarterStr) {
  const match = quarterStr.match(/Q(\d) (\d{4})/);
  if (!match) return null;
  const quarter = parseInt(match[1]);
  const year = parseInt(match[2]);
  return { quarter, year };
}

function generateQuarters(startQuarter, numQuarters) {
  const parsed = parseQuarter(startQuarter);
  if (!parsed) return [];
  
  const quarters = [];
  let { quarter, year } = parsed;
  
  for (let i = 0; i < numQuarters; i++) {
    quarters.push(`Q${quarter} ${year}`);
    quarter++;
    if (quarter > 4) {
      quarter = 1;
      year++;
    }
  }
  
  return quarters;
}

// Safe Gompertz model functions with numerical stability
function gompertzModel(t, K, b, t0) {
  // Validate inputs
  if (!isFinite(K) || !isFinite(b) || !isFinite(t0)) {
    console.warn('Invalid Gompertz parameters:', { K, b, t0 });
    return NaN;
  }
  if (K <= 0 || b <= 0) {
    console.warn('K and b must be positive:', { K, b });
    return NaN;
  }
  
  // Calculate with overflow protection
  let exponent = -b * (t - t0);
  // Clamp to safe range to prevent overflow
  exponent = Math.max(-700, Math.min(700, exponent));
  
  let inner_exp = Math.exp(exponent);
  if (!isFinite(inner_exp)) {
    console.warn('Inner exp overflow at t=' + t);
    return NaN;
  }
  
  let outer_exp = -inner_exp;
  // Clamp again for safety
  outer_exp = Math.max(-700, Math.min(700, outer_exp));
  
  let result = K * Math.exp(outer_exp);
  
  if (!isFinite(result)) {
    console.warn('Final result not finite at t=' + t);
    return NaN;
  }
  
  return result;
}

function fitGompertzCurve(data) {
  console.log('Starting Gompertz curve fitting...');
  const n = data.length;
  const maxAccounts = Math.max(...data);
  const minAccounts = Math.min(...data);
  
  console.log('Data range:', { min: minAccounts, max: maxAccounts, points: n });
  
  let bestParams = null;
  let bestError = Infinity;
  let validFits = 0;
  
  // Robust grid search with validation
  const K_steps = [1.1, 1.3, 1.5, 2.0, 2.5];
  const b_steps = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4];
  const t0_steps = [-2, 0, n * 0.3, n * 0.5, n * 0.7, n + 2];
  
  for (const k_mult of K_steps) {
    const K = maxAccounts * k_mult;
    if (K <= maxAccounts) continue; // K must be greater than observed max
    
    for (const b of b_steps) {
      if (b <= 0 || b > 1.0) continue; // b must be positive and reasonable
      
      for (const t0 of t0_steps) {
        let error = 0;
        let hasNaN = false;
        
        for (let i = 0; i < n; i++) {
          const predicted = gompertzModel(i, K, b, t0);
          if (!isFinite(predicted)) {
            hasNaN = true;
            break;
          }
          error += Math.pow(data[i] - predicted, 2);
        }
        
        if (!hasNaN && isFinite(error) && error < bestError) {
          bestError = error;
          bestParams = { K, b, t0 };
          validFits++;
        }
      }
    }
  }
  
  console.log('Valid fits found:', validFits);
  
  if (!bestParams) {
    console.error('No valid fit found! Using fallback parameters.');
    // Fallback: simple parameters
    bestParams = {
      K: maxAccounts * 1.5,
      b: 0.2,
      t0: n * 0.5
    };
  }
  
  console.log('Best initial params:', bestParams);
  
  // Careful refinement with bounds checking
  const learningRate = 0.001; // Reduced for stability
  const iterations = 50; // Fewer iterations
  
  for (let iter = 0; iter < iterations; iter++) {
    let gradK = 0, gradB = 0, gradT0 = 0;
    let validGrads = true;
    
    for (let i = 0; i < n; i++) {
      const predicted = gompertzModel(i, bestParams.K, bestParams.b, bestParams.t0);
      if (!isFinite(predicted)) {
        validGrads = false;
        break;
      }
      
      const error = predicted - data[i];
      
      const exponent = -bestParams.b * (i - bestParams.t0);
      if (Math.abs(exponent) > 100) continue; // Skip if too large
      
      const exp_inner = Math.exp(exponent);
      const exp_outer = Math.exp(-exp_inner);
      
      if (!isFinite(exp_inner) || !isFinite(exp_outer)) {
        validGrads = false;
        break;
      }
      
      gradK += 2 * error * exp_outer;
      gradB += 2 * error * bestParams.K * exp_outer * exp_inner * (i - bestParams.t0);
      gradT0 += 2 * error * bestParams.K * exp_outer * exp_inner * bestParams.b;
    }
    
    if (!validGrads || !isFinite(gradK) || !isFinite(gradB) || !isFinite(gradT0)) {
      console.warn('Invalid gradients at iteration', iter);
      break;
    }
    
    // Update with bounds checking
    const newK = bestParams.K - learningRate * gradK / n;
    const newB = bestParams.b - learningRate * gradB / n;
    const newT0 = bestParams.t0 - learningRate * gradT0 / n;
    
    // Validate new parameters
    if (newK > maxAccounts && newK < maxAccounts * 5 && 
        newB > 0.01 && newB < 1.0 &&
        isFinite(newK) && isFinite(newB) && isFinite(newT0)) {
      bestParams.K = newK;
      bestParams.b = newB;
      bestParams.t0 = newT0;
    } else {
      console.warn('Parameter update rejected at iteration', iter);
      break;
    }
  }
  
  console.log('Final fitted params:', bestParams);
  
  // Validate final parameters
  if (!isFinite(bestParams.K) || !isFinite(bestParams.b) || !isFinite(bestParams.t0)) {
    console.error('Final parameters contain NaN/Infinity!');
    alert('擬合失敗：參數無效。請檢查數據質量。');
    return null;
  }
  
  if (bestParams.K <= maxAccounts) {
    console.warn('K is not greater than max data, adjusting...');
    bestParams.K = maxAccounts * 1.2;
  }
  
  if (bestParams.b <= 0 || bestParams.b > 1.0) {
    console.warn('b is out of range, clamping...');
    bestParams.b = Math.max(0.05, Math.min(0.5, bestParams.b));
  }
  
  // Calculate R-squared with validation
  const mean = data.reduce((sum, val) => sum + val, 0) / n;
  let ssRes = 0, ssTot = 0;
  let hasNaNPredictions = false;
  
  for (let i = 0; i < n; i++) {
    const predicted = gompertzModel(i, bestParams.K, bestParams.b, bestParams.t0);
    if (!isFinite(predicted)) {
      hasNaNPredictions = true;
      console.error('NaN prediction at index', i);
      continue;
    }
    ssRes += Math.pow(data[i] - predicted, 2);
    ssTot += Math.pow(data[i] - mean, 2);
  }
  
  if (hasNaNPredictions) {
    console.error('Some predictions are NaN!');
  }
  
  const r2 = ssTot > 0 ? Math.max(0, Math.min(1, 1 - (ssRes / ssTot))) : 0;
  const rmse = Math.sqrt(ssRes / n);
  
  console.log('Fit quality:', { r2, rmse });
  
  if (!isFinite(r2) || r2 < 0 || r2 > 1) {
    console.warn('Invalid R²:', r2);
  }
  
  return {
    params: bestParams,
    r2: isFinite(r2) ? r2 : 0,
    rmse: isFinite(rmse) ? rmse : Infinity,
    valid: !hasNaNPredictions && isFinite(r2)
  };
}

function applyIntervention(t, baselineValue, params, interventionParams, launchQuarter) {
  // Validate inputs
  if (!isFinite(t) || !isFinite(baselineValue)) {
    console.warn('Invalid intervention inputs:', { t, baselineValue });
    return baselineValue;
  }
  
  const { K, b, t0 } = params;
  const { alpha, delta_t, kappa, half_life, window_length, expansion_length } = interventionParams;
  
  // Validate parameters
  if (!isFinite(K) || !isFinite(b) || !isFinite(t0)) {
    console.warn('Invalid params in intervention:', params);
    return baselineValue;
  }
  
  if (t < launchQuarter) {
    return baselineValue;
  }
  
  const tSinceLaunch = t - launchQuarter;
  
  // Growth acceleration with exponential decay (with safety bounds)
  let accelerationFactor = 1;
  if (tSinceLaunch < window_length) {
    const decayExponent = -Math.log(2) * tSinceLaunch / half_life;
    // Clamp decay exponent
    const clampedExp = Math.max(-100, Math.min(0, decayExponent));
    const decayFactor = Math.exp(clampedExp);
    
    if (!isFinite(decayFactor)) {
      console.warn('Invalid decay factor at t=' + t);
      accelerationFactor = 1;
    } else {
      // Cap alpha effect
      const cappedAlpha = Math.min(0.5, Math.max(0, alpha));
      accelerationFactor = 1 + cappedAlpha * decayFactor;
    }
  }
  
  // TAM expansion with smooth transition (with safety bounds)
  let capacityFactor = 1;
  if (tSinceLaunch < expansion_length) {
    const expansionProgress = tSinceLaunch / expansion_length;
    const expansionExp = -3 * expansionProgress;
    // Clamp expansion exponent
    const clampedExpansion = Math.max(-100, Math.min(0, expansionExp));
    const expansionSmoother = 1 - Math.exp(clampedExpansion);
    
    if (!isFinite(expansionSmoother)) {
      console.warn('Invalid expansion smoother at t=' + t);
      capacityFactor = 1;
    } else {
      // Cap kappa effect
      const cappedKappa = Math.min(0.2, Math.max(0, kappa));
      capacityFactor = 1 + cappedKappa * expansionSmoother;
    }
  } else {
    const cappedKappa = Math.min(0.2, Math.max(0, kappa));
    capacityFactor = 1 + cappedKappa;
  }
  
  // Validate factors
  if (!isFinite(accelerationFactor) || !isFinite(capacityFactor)) {
    console.warn('Invalid factors:', { accelerationFactor, capacityFactor });
    return baselineValue;
  }
  
  // Apply intervention with bounds
  const adjustedK = K * capacityFactor;
  const adjustedB = b * Math.min(2.0, accelerationFactor); // Cap acceleration
  const adjustedT0 = t0 - Math.min(5, Math.max(0, delta_t)); // Cap shift
  
  // Validate adjusted parameters
  if (!isFinite(adjustedK) || !isFinite(adjustedB) || !isFinite(adjustedT0)) {
    console.warn('Invalid adjusted params:', { adjustedK, adjustedB, adjustedT0 });
    return baselineValue;
  }
  
  const result = gompertzModel(t, adjustedK, adjustedB, adjustedT0);
  
  // Return baseline if result is invalid
  if (!isFinite(result)) {
    console.warn('Invalid intervention result at t=' + t + ', returning baseline');
    return baselineValue;
  }
  
  return result;
}

// Step navigation
function goToStep(step) {
  // Hide all sections
  for (let i = 1; i <= 4; i++) {
    document.getElementById(`step${i}`).style.display = 'none';
    const stepEl = document.querySelector(`.step[data-step="${i}"]`);
    stepEl.classList.remove('active', 'completed');
    if (i < step) {
      stepEl.classList.add('completed');
    }
  }
  
  // Show current section
  document.getElementById(`step${step}`).style.display = 'block';
  document.querySelector(`.step[data-step="${step}"]`).classList.add('active');
  state.currentStep = step;
}

// Step 1: Data Input
function loadSampleData() {
  // Create sample data with some missing values to demonstrate the feature
  const dataText = SAMPLE_DATA.periods.map((period, i) => {
    // Mark Q2 2022 and Q1 2023 as missing for demonstration
    if (i === 9 || i === 12) {
      return `${period},-`;
    }
    return `${period},${SAMPLE_DATA.accounts[i]}`;
  }).join('\n');
  
  document.getElementById('dataInput').value = dataText;
  parseAndDisplayData();
}

function parseAndDisplayData() {
  const dataInput = document.getElementById('dataInput').value;
  const lines = dataInput.trim().split('\n');
  
  const data = [];
  const errorEl = document.getElementById('dataError');
  
  try {
    for (const line of lines) {
      if (!line.trim()) continue;
      
      const parts = line.split(',');
      if (parts.length !== 2) {
        throw new Error('每行必須包含兩個值: 期間,帳戶數');
      }
      
      const period = parts[0].trim();
      const accountsStr = parts[1].trim();
      
      // Handle missing data markers
      let accounts = null;
      let isMissing = false;
      
      if (accountsStr === '' || accountsStr === '-' || accountsStr.toLowerCase() === 'null' || accountsStr.toLowerCase() === 'na') {
        isMissing = true;
      } else {
        accounts = parseFloat(accountsStr);
        if (isNaN(accounts)) {
          throw new Error(`無效的帳戶數: ${accountsStr}`);
        }
        if (accounts < 0) {
          throw new Error(`帳戶數不能為負數: ${accountsStr}`);
        }
      }
      
      data.push({ period, accounts, isMissing, dataType: isMissing ? 'Missing' : 'Original' });
    }
    
    if (data.length < 8) {
      throw new Error('至少需要 8 個數據點進行擬合');
    }
    
    state.historicalData = data;
    
    // Perform data quality analysis
    analyzeDataQuality();
    displayDataQualityReport();
    displayDataPreview(data);
    
    errorEl.style.display = 'none';
    
  } catch (error) {
    errorEl.textContent = `錯誤: ${error.message}`;
    errorEl.style.display = 'block';
  }
}

// Data Quality Analysis
function analyzeDataQuality() {
  const data = state.historicalData;
  const issues = [];
  let score = 100;
  
  // Check for missing data
  const missingCount = data.filter(d => d.isMissing).length;
  const missingPercent = (missingCount / data.length) * 100;
  
  if (missingCount === 0) {
    issues.push({
      type: 'success',
      icon: '✓',
      title: '數據完整性: 優異',
      description: `所有 ${data.length} 個季度的數據都存在`
    });
  } else if (missingCount <= 2) {
    issues.push({
      type: 'warning',
      icon: '⚠️',
      title: `警告: ${missingCount} 個季度缺漏`,
      description: '建議使用插值方法填補缺漏值'
    });
    score -= 10 * missingCount;
  } else if (missingCount <= 5) {
    issues.push({
      type: 'warning',
      icon: '⚠️',
      title: `警告: ${missingCount} 個季度缺漏 (${missingPercent.toFixed(1)}%)`,
      description: '缺漏較多，建議使用 Gompertz 插值並驗證結果'
    });
    score -= 30;
  } else {
    issues.push({
      type: 'error',
      icon: '❌',
      title: `嚴重: ${missingCount} 個季度缺漏 (${missingPercent.toFixed(1)}%)`,
      description: '數據缺漏過多，可能影響模型可靠性。建議嘗試獲取實際數據。'
    });
    score -= 50;
  }
  
  // Check for outliers (only on non-missing data)
  const validData = data.filter(d => !d.isMissing);
  if (validData.length >= 5) {
    const accounts = validData.map(d => d.accounts);
    const mean = accounts.reduce((sum, v) => sum + v, 0) / accounts.length;
    const stdDev = Math.sqrt(accounts.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / accounts.length);
    
    let outlierCount = 0;
    validData.forEach((d, i) => {
      const zScore = Math.abs((d.accounts - mean) / stdDev);
      if (zScore > 2) {
        d.isOutlier = true;
        outlierCount++;
      }
    });
    
    if (outlierCount > 0) {
      issues.push({
        type: 'warning',
        icon: '⚠️',
        title: `檢測到 ${outlierCount} 個異常值`,
        description: '這些值與平均趨勢差異較大，請在數據表中確認是否為真實數據'
      });
      score -= 5 * outlierCount;
    }
  }
  
  // Check for negative growth (non-monotonic)
  const nonMonotonicCount = validData.filter((d, i) => {
    if (i === 0) return false;
    const prev = validData[i - 1];
    return d.accounts < prev.accounts;
  }).length;
  
  if (nonMonotonicCount > 0) {
    issues.push({
      type: 'warning',
      icon: '⚠️',
      title: `檢測到 ${nonMonotonicCount} 個負增長季度`,
      description: 'Gompertz 模型假設單調增長，負增長可能影響擬合品質'
    });
    score -= 5 * nonMonotonicCount;
  }
  
  // Check minimum data points
  const validCount = validData.length;
  if (validCount < 8) {
    issues.push({
      type: 'error',
      icon: '❌',
      title: '有效數據點不足',
      description: `僅有 ${validCount} 個有效數據點，至少需要 8 個才能可靠擬合 Gompertz 曲線`
    });
    score -= 40;
  } else if (validCount < 12) {
    issues.push({
      type: 'warning',
      icon: '⚠️',
      title: '有效數據點較少',
      description: `僅有 ${validCount} 個有效數據點，建議至少 12 個以獲得更可靠的預測`
    });
    score -= 10;
  }
  
  score = Math.max(0, Math.min(100, score));
  
  state.dataQuality = {
    score,
    issues,
    missingCount,
    validCount,
    totalCount: data.length
  };
}

function displayDataQualityReport() {
  const card = document.getElementById('dataQualityCard');
  const quality = state.dataQuality;
  
  // Update score
  const scoreFill = document.getElementById('qualityScoreFill');
  const scoreValue = document.getElementById('qualityScoreValue');
  
  scoreFill.style.width = quality.score + '%';
  scoreValue.textContent = quality.score + '/100';
  
  // Determine quality level
  let qualityClass = 'poor';
  let qualityLabel = '⚠️';
  if (quality.score >= 95) {
    qualityClass = 'excellent';
    qualityLabel = '✓';
  } else if (quality.score >= 80) {
    qualityClass = 'good';
    qualityLabel = '✓';
  } else if (quality.score >= 60) {
    qualityClass = 'acceptable';
    qualityLabel = '⚠️';
  } else {
    qualityClass = 'poor';
    qualityLabel = '❌';
  }
  
  scoreFill.className = 'quality-score-fill ' + qualityClass;
  scoreValue.innerHTML = qualityLabel + ' ' + quality.score + '/100';
  
  // Display issues
  const issuesContainer = document.getElementById('qualityIssues');
  issuesContainer.innerHTML = quality.issues.map(issue => `
    <div class="issue-item ${issue.type}">
      <div class="issue-icon">${issue.icon}</div>
      <div class="issue-content">
        <div class="issue-title">${issue.title}</div>
        <div class="issue-description">${issue.description}</div>
      </div>
    </div>
  `).join('');
  
  // Display recommendations
  const recommendations = [];
  
  if (quality.missingCount > 0) {
    if (quality.missingCount <= 2) {
      recommendations.push('使用 Gompertz 插值填補缺漏值（推薦）');
    } else if (quality.missingCount <= 5) {
      recommendations.push('使用 Gompertz 插值填補缺漏值，並在擬合後驗證參數穩定性');
      recommendations.push('如果可能，嘗試從數據源獲取實際缺漏值');
    } else {
      recommendations.push('優先嘗試獲取實際缺漏數據');
      recommendations.push('如無法獲取，建議縮短分析期間至數據較完整的區段');
    }
  }
  
  if (quality.score < 80) {
    recommendations.push('預測結果的不確定性將會較高，請謹慎使用');
    recommendations.push('建議在最終報告中標註數據品質問題');
  }
  
  if (recommendations.length > 0) {
    const recContainer = document.getElementById('qualityRecommendations');
    recContainer.innerHTML = `
      <h4>📊 建議行動</h4>
      <ul>
        ${recommendations.map(r => '<li>' + r + '</li>').join('')}
      </ul>
    `;
  }
  
  card.style.display = 'block';
  
  // Show missing data handling card if there are missing values
  if (quality.missingCount > 0) {
    document.getElementById('missingDataCard').style.display = 'block';
    document.getElementById('bestPracticesCard').style.display = 'block';
  } else {
    document.getElementById('missingDataCard').style.display = 'none';
    document.getElementById('bestPracticesCard').style.display = 'none';
  }
}

function displayDataPreview(data) {
  const tbody = document.querySelector('#dataPreviewTable tbody');
  tbody.innerHTML = '';
  
  data.forEach((row, index) => {
    const tr = document.createElement('tr');
    
    // Determine row status
    let statusClass = 'status-valid';
    let statusBadge = '<span class="status-badge valid">✓ 正常</span>';
    let outlierBadge = '-';
    let recommendation = '無需處理';
    let actions = '-';
    
    if (row.isMissing) {
      statusClass = 'status-missing';
      statusBadge = '<span class="status-badge missing">⚠️ 缺漏</span>';
      recommendation = '建議插值';
      actions = '<button class="btn btn--sm btn--outline" onclick="editCell(' + index + ')">手動輸入</button>';
    } else if (row.dataType === 'Interpolated') {
      statusClass = 'status-interpolated';
      statusBadge = '<span class="status-badge interpolated">📊 已插值</span>';
      recommendation = '已處理';
      actions = '<button class="btn btn--sm btn--outline" onclick="editCell(' + index + ')">修改</button>';
    } else if (row.isOutlier) {
      statusClass = 'status-outlier';
      outlierBadge = '<span class="status-badge outlier">⚠️ 異常</span>';
      recommendation = '請確認數據';
      actions = '<button class="btn btn--sm btn--outline" onclick="confirmCell(' + index + ')">確認無誤</button>';
    }
    
    const valueDisplay = row.isMissing ? '<em style="color: var(--color-text-secondary);">缺漏</em>' : formatNumber(row.accounts);
    
    tr.className = statusClass;
    tr.innerHTML = `
      <td>${row.period}</td>
      <td>${valueDisplay}</td>
      <td>${statusBadge}</td>
      <td>${outlierBadge}</td>
      <td style="font-size: var(--font-size-sm);">${recommendation}</td>
      <td class="cell-actions">${actions}</td>
    `;
    tbody.appendChild(tr);
  });
  
  document.getElementById('dataPreviewCard').style.display = 'block';
}

// Cell editing functions
function editCell(index) {
  const value = prompt('請輸入帳戶數:', state.historicalData[index].accounts || '');
  if (value !== null) {
    const accounts = parseFloat(value);
    if (!isNaN(accounts) && accounts >= 0) {
      state.historicalData[index].accounts = accounts;
      state.historicalData[index].isMissing = false;
      state.historicalData[index].dataType = 'Corrected';
      analyzeDataQuality();
      displayDataQualityReport();
      displayDataPreview(state.historicalData);
    } else {
      alert('請輸入有效的數字');
    }
  }
}

function confirmCell(index) {
  state.historicalData[index].isOutlier = false;
  state.historicalData[index].dataType = 'Confirmed';
  analyzeDataQuality();
  displayDataQualityReport();
  displayDataPreview(state.historicalData);
}

function validateAndProceed() {
  if (state.historicalData.length === 0) {
    alert('請先輸入或載入數據');
    return;
  }
  
  // Check data quality
  if (!state.dataQuality) {
    analyzeDataQuality();
  }
  
  // Warn if quality is poor
  if (state.dataQuality.score < 60) {
    if (!confirm('數據品質分數較低 (' + state.dataQuality.score + '/100)。預測結果可能不可靠。確定要繼續嗎？')) {
      return;
    }
  }
  
  // Check if there are still missing values
  const stillMissing = state.historicalData.filter(d => d.isMissing).length;
  if (stillMissing > 0) {
    if (state.interpolationMethod === 'ignore' || !state.interpolationMethod) {
      if (!confirm(`仍有 ${stillMissing} 個缺漏值未處理。將在擬合時忽略這些點。確定要繼續嗎？`)) {
        return;
      }
    }
  }
  
  goToStep(2);
  performFitting();
}

// Step 2: Baseline Fitting
function performFitting() {
  const statusEl = document.getElementById('fittingStatus');
  const paramsEl = document.getElementById('parametersDisplay');
  
  statusEl.textContent = '正在擬合模型...';
  statusEl.className = 'status-message';
  paramsEl.style.display = 'none';
  
  setTimeout(() => {
    // Only use non-missing data for fitting
    const validData = state.historicalData.filter(d => !d.isMissing);
    const accounts = validData.map(d => d.accounts);
    const result = fitGompertzCurve(accounts);
    
    if (!result) {
      statusEl.textContent = '✘ 擬合失敗：無法找到有效參數。請檢查數據是否遵循S型曲線模式。';
      statusEl.style.color = 'var(--color-error)';
      statusEl.style.background = 'rgba(var(--color-error-rgb), 0.1)';
      return;
    }
    
    state.fittedParams = result.params;
    
    // Validate and display parameters with status indicators
    const paramKEl = document.getElementById('paramK');
    const paramBEl = document.getElementById('paramB');
    const paramT0El = document.getElementById('paramT0');
    
    // Calculate confidence based on data quality
    const dataQualityScore = state.dataQuality ? state.dataQuality.score : 100;
    let confidenceLevel = 'high';
    let confidenceColor = 'var(--color-success)';
    let confidenceLabel = '高信心度';
    
    if (dataQualityScore >= 95) {
      confidenceLevel = 'high';
      confidenceColor = 'var(--color-success)';
      confidenceLabel = '✓ 高信心度 (數據品質優異)';
    } else if (dataQualityScore >= 80) {
      confidenceLevel = 'moderate';
      confidenceColor = 'var(--color-primary)';
      confidenceLabel = '⚠️ 中等信心度 (數據品質良好)';
    } else if (dataQualityScore >= 60) {
      confidenceLevel = 'low';
      confidenceColor = 'var(--color-warning)';
      confidenceLabel = '⚠️ 低信心度 (數據品質可接受)';
    } else {
      confidenceLevel = 'very-low';
      confidenceColor = 'var(--color-error)';
      confidenceLabel = '❌ 極低信心度 (數據品質差)';
    }
    
    const K = result.params.K;
    const b = result.params.b;
    const t0 = result.params.t0;
    
    // Check parameter validity
    const maxData = Math.max(...accounts);
    const K_valid = isFinite(K) && K > maxData;
    const b_valid = isFinite(b) && b > 0.01 && b < 1.0;
    const t0_valid = isFinite(t0);
    const r2_valid = isFinite(result.r2) && result.r2 >= 0 && result.r2 <= 1;
    
    // Display with validation indicators
    paramKEl.innerHTML = (K_valid ? '✔ ' : '⚠️ ') + formatNumber(K);
    paramKEl.style.color = K_valid ? 'var(--color-success)' : 'var(--color-warning)';
    
    paramBEl.innerHTML = (b_valid ? '✔ ' : '⚠️ ') + b.toFixed(4);
    paramBEl.style.color = b_valid ? 'var(--color-success)' : 'var(--color-warning)';
    
    paramT0El.innerHTML = (t0_valid ? '✔ ' : '⚠️ ') + t0.toFixed(2);
    paramT0El.style.color = t0_valid ? 'var(--color-success)' : 'var(--color-warning)';
    
    document.getElementById('paramK_ci').textContent = '承載容量上限 (K > ' + formatNumber(maxData) + ')';
    document.getElementById('paramB_ci').textContent = '成長速率參數 (合理範圍: 0.01-1.0)';
    document.getElementById('paramT0_ci').textContent = '轉折點時間 (相對於起始點)';
    
    const r2Display = r2_valid ? result.r2.toFixed(4) : 'N/A';
    const r2Color = r2_valid && result.r2 > 0.95 ? 'var(--color-success)' : 
                     r2_valid && result.r2 > 0.85 ? 'var(--color-warning)' : 'var(--color-error)';
    
    const metricR2El = document.getElementById('metricR2');
    metricR2El.textContent = r2Display;
    metricR2El.style.color = r2Color;
    
    document.getElementById('metricRMSE').textContent = isFinite(result.rmse) ? formatNumber(result.rmse) : 'N/A';
    
    // Show quality message
    if (result.valid && r2_valid) {
      if (result.r2 > 0.95) {
        statusEl.innerHTML = '✔ 擬合成功！R² = ' + result.r2.toFixed(4) + ' (優異)<br><span style="color: ' + confidenceColor + '; font-size: var(--font-size-sm);">' + confidenceLabel + '</span>';
        statusEl.style.color = 'var(--color-success)';
        statusEl.style.background = 'rgba(var(--color-success-rgb), 0.1)';
      } else if (result.r2 > 0.85) {
        statusEl.innerHTML = '⚠️ 擬合完成，R² = ' + result.r2.toFixed(4) + ' (品質中等，可能有雜訊)<br><span style="color: ' + confidenceColor + '; font-size: var(--font-size-sm);">' + confidenceLabel + '</span>';
        statusEl.style.color = 'var(--color-warning)';
        statusEl.style.background = 'rgba(var(--color-warning-rgb), 0.1)';
      } else {
        statusEl.innerHTML = '⚠️ 擬合完成，R² = ' + result.r2.toFixed(4) + ' (品質較低，建議檢查數據)<br><span style="color: ' + confidenceColor + '; font-size: var(--font-size-sm);">' + confidenceLabel + '</span>';
        statusEl.style.color = 'var(--color-warning)';
        statusEl.style.background = 'rgba(var(--color-warning-rgb), 0.1)';
      }
    } else {
      statusEl.innerHTML = '⚠️ 擬合完成但參數可能不穩定，請謹慎使用預測結果<br><span style="color: ' + confidenceColor + '; font-size: var(--font-size-sm);">' + confidenceLabel + '</span>';
      statusEl.style.color = 'var(--color-error)';
      statusEl.style.background = 'rgba(var(--color-error-rgb), 0.1)';
    }
    
    statusEl.style.display = 'block';
    paramsEl.style.display = 'block';
    
    displayFittingChart();
  }, 500);
}

function displayFittingChart() {
  const ctx = document.getElementById('fittingChart').getContext('2d');
  
  const periods = state.historicalData.map(d => d.period);
  const actual = state.historicalData.map(d => d.accounts);
  const fitted = state.historicalData.map((d, i) => 
    gompertzModel(i, state.fittedParams.K, state.fittedParams.b, state.fittedParams.t0)
  );
  
  if (fittingChart) {
    fittingChart.destroy();
  }
  
  fittingChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: periods,
      datasets: [
        {
          label: '實際數據',
          data: actual,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          pointRadius: 5,
          pointHoverRadius: 7,
          borderWidth: 2
        },
        {
          label: '擬合曲線',
          data: fitted,
          borderColor: '#10b981',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top'
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return context.dataset.label + ': ' + formatNumber(context.parsed.y);
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            callback: function(value) {
              return formatNumber(value);
            }
          }
        }
      }
    }
  });
}

// Step 3: Scenario Configuration
function initializeSliders() {
  const scenarios = ['conservative', 'moderate', 'aggressive'];
  const params = ['alpha', 'delta_t', 'kappa', 'half_life', 'window', 'expansion'];
  
  scenarios.forEach(scenario => {
    params.forEach(param => {
      const sliderId = `${param}_${scenario}`;
      const slider = document.getElementById(sliderId);
      const valueEl = document.getElementById(`${sliderId}_val`);
      
      if (slider && valueEl) {
        slider.addEventListener('input', (e) => {
          const value = parseFloat(e.target.value);
          valueEl.textContent = value;
          
          // Update state
          const paramKey = param === 'window' ? 'window_length' : param === 'expansion' ? 'expansion_length' : param;
          state.scenarioParams[scenario][paramKey] = value;
        });
      }
    });
  });
}

function loadScenarioPreset(scenario) {
  const presets = {
    conservative: { alpha: 0.10, delta_t: 0.5, kappa: 0.02, half_life: 6, window_length: 6, expansion_length: 8 },
    moderate: { alpha: 0.20, delta_t: 1.0, kappa: 0.05, half_life: 8, window_length: 8, expansion_length: 10 },
    aggressive: { alpha: 0.35, delta_t: 1.5, kappa: 0.08, half_life: 10, window_length: 12, expansion_length: 12 }
  };
  
  const preset = presets[scenario];
  state.scenarioParams[scenario] = { ...preset };
  
  // Update sliders
  document.getElementById(`alpha_${scenario}`).value = preset.alpha;
  document.getElementById(`alpha_${scenario}_val`).textContent = preset.alpha;
  
  document.getElementById(`delta_t_${scenario}`).value = preset.delta_t;
  document.getElementById(`delta_t_${scenario}_val`).textContent = preset.delta_t;
  
  document.getElementById(`kappa_${scenario}`).value = preset.kappa;
  document.getElementById(`kappa_${scenario}_val`).textContent = preset.kappa;
  
  document.getElementById(`half_life_${scenario}`).value = preset.half_life;
  document.getElementById(`half_life_${scenario}_val`).textContent = preset.half_life;
  
  document.getElementById(`window_${scenario}`).value = preset.window_length;
  document.getElementById(`window_${scenario}_val`).textContent = preset.window_length;
  
  document.getElementById(`expansion_${scenario}`).value = preset.expansion_length;
  document.getElementById(`expansion_${scenario}_val`).textContent = preset.expansion_length;
}

// Step 4: Results & Analysis
function generateForecasts() {
  const forecastYears = parseInt(document.getElementById('forecastYears').value);
  const forecastQuarters = forecastYears * 4;
  const platformLaunch = parseInt(document.getElementById('platformLaunch').value);
  
  state.platformLaunchQuarter = platformLaunch;
  
  console.log('Generating forecasts for', forecastQuarters, 'quarters...');
  console.log('Platform launch at quarter:', platformLaunch);
  console.log('Using fitted params:', state.fittedParams);
  
  const lastPeriod = state.historicalData[state.historicalData.length - 1].period;
  const forecastPeriods = generateQuarters(lastPeriod, forecastQuarters + 1).slice(1);
  
  const baseOffset = state.historicalData.length;
  
  const forecasts = {
    periods: forecastPeriods,
    baseline: [],
    conservative: [],
    moderate: [],
    aggressive: []
  };
  
  let hasInvalidForecasts = false;
  
  for (let i = 0; i < forecastQuarters; i++) {
    const t = baseOffset + i;
    const baseline = gompertzModel(t, state.fittedParams.K, state.fittedParams.b, state.fittedParams.t0);
    
    if (!isFinite(baseline)) {
      console.error('Invalid baseline at t=' + t);
      hasInvalidForecasts = true;
      forecasts.baseline.push(NaN);
      forecasts.conservative.push(NaN);
      forecasts.moderate.push(NaN);
      forecasts.aggressive.push(NaN);
      continue;
    }
    
    forecasts.baseline.push(baseline);
    
    // Calculate uncertainty based on data quality
    const uncertaintyFactor = state.dataQuality ? (100 - state.dataQuality.score) / 100 : 0;
    const baselineUncertainty = baseline * (0.02 + uncertaintyFactor * 0.06); // 2-8% uncertainty
    
    const conservative = applyIntervention(t, baseline, state.fittedParams, state.scenarioParams.conservative, baseOffset + platformLaunch);
    const moderate = applyIntervention(t, baseline, state.fittedParams, state.scenarioParams.moderate, baseOffset + platformLaunch);
    const aggressive = applyIntervention(t, baseline, state.fittedParams, state.scenarioParams.aggressive, baseOffset + platformLaunch);
    
    if (!isFinite(conservative) || !isFinite(moderate) || !isFinite(aggressive)) {
      console.error('Invalid intervention result at t=' + t);
      hasInvalidForecasts = true;
    }
    
    forecasts.conservative.push(conservative);
    forecasts.moderate.push(moderate);
    forecasts.aggressive.push(aggressive);
  }
  
  if (hasInvalidForecasts) {
    console.warn('Some forecast values are invalid!');
    alert('⚠️ 警告：某些預測值無效。請檢查結果表格中的紅色標記。');
  }
  
  console.log('Forecast generation complete. Sample baseline:', forecasts.baseline.slice(0, 3));
  console.log('Sample conservative:', forecasts.conservative.slice(0, 3));
  
  // Add confidence bands based on data quality
  const dataQualityScore = state.dataQuality ? state.dataQuality.score : 100;
  const uncertaintyPercent = dataQualityScore >= 95 ? 2 : dataQualityScore >= 80 ? 5 : 8;
  
  forecasts.uncertaintyPercent = uncertaintyPercent;
  state.forecastData = forecasts;
  
  goToStep(4);
  displayForecastChart();
  displayIncrementalChart();
  displayForecastTable();
  displaySummaryStats();
}

function displayForecastChart() {
  const ctx = document.getElementById('forecastChart').getContext('2d');
  
  if (forecastChart) {
    forecastChart.destroy();
  }
  
  // Filter out NaN values for charting
  const filterNaN = (arr) => arr.map(v => isFinite(v) ? v : null);
  
  const launchIndex = state.platformLaunchQuarter - 1;
  
  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: state.forecastData.periods,
      datasets: [
        {
          label: '基線',
          data: filterNaN(state.forecastData.baseline),
          borderColor: '#6b7280',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          spanGaps: false
        },
        {
          label: '保守型',
          data: filterNaN(state.forecastData.conservative),
          borderColor: '#3b82f6',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          spanGaps: false
        },
        {
          label: '穩健型',
          data: filterNaN(state.forecastData.moderate),
          borderColor: '#10b981',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          spanGaps: false
        },
        {
          label: '積極型',
          data: filterNaN(state.forecastData.aggressive),
          borderColor: '#ef4444',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          spanGaps: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top'
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const value = context.parsed.y;
              if (value === null || !isFinite(value)) {
                return context.dataset.label + ': N/A';
              }
              return context.dataset.label + ': ' + formatNumber(value);
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: false,
          ticks: {
            callback: function(value) {
              return formatNumber(value);
            }
          }
        }
      }
    }
  });
}

function displayIncrementalChart() {
  const ctx = document.getElementById('incrementalChart').getContext('2d');
  
  // Safe incremental calculation
  const safeIncremental = (scenario) => {
    return state.forecastData[scenario].map((v, i) => {
      const baseline = state.forecastData.baseline[i];
      if (isFinite(v) && isFinite(baseline)) {
        return v - baseline;
      }
      return null;
    });
  };
  
  const incrementalConservative = safeIncremental('conservative');
  const incrementalModerate = safeIncremental('moderate');
  const incrementalAggressive = safeIncremental('aggressive');
  
  if (incrementalChart) {
    incrementalChart.destroy();
  }
  
  incrementalChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: state.forecastData.periods,
      datasets: [
        {
          label: '保守型增量',
          data: incrementalConservative,
          backgroundColor: 'rgba(59, 130, 246, 0.6)',
          borderColor: '#3b82f6',
          borderWidth: 1
        },
        {
          label: '穩健型增量',
          data: incrementalModerate,
          backgroundColor: 'rgba(16, 185, 129, 0.6)',
          borderColor: '#10b981',
          borderWidth: 1
        },
        {
          label: '積極型增量',
          data: incrementalAggressive,
          backgroundColor: 'rgba(239, 68, 68, 0.6)',
          borderColor: '#ef4444',
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top'
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const value = context.parsed.y;
              if (value === null || !isFinite(value)) {
                return context.dataset.label + ': N/A';
              }
              return context.dataset.label + ': ' + formatNumber(value);
            }
          }
        }
      },
      scales: {
        x: {
          stacked: false
        },
        y: {
          stacked: false,
          beginAtZero: true,
          ticks: {
            callback: function(value) {
              return formatNumber(value);
            }
          }
        }
      }
    }
  });
}

function displayForecastTable() {
  const tbody = document.querySelector('#forecastTable tbody');
  tbody.innerHTML = '';
  
  const n = state.forecastData.periods.length;
  let hasInvalidValues = false;
  
  for (let i = 0; i < n; i++) {
    const tr = document.createElement('tr');
    
    const baseline = state.forecastData.baseline[i];
    const conservative = state.forecastData.conservative[i];
    const moderate = state.forecastData.moderate[i];
    const aggressive = state.forecastData.aggressive[i];
    
    // Check for invalid values
    const baselineValid = isFinite(baseline);
    const conservativeValid = isFinite(conservative);
    const moderateValid = isFinite(moderate);
    const aggressiveValid = isFinite(aggressive);
    
    if (!baselineValid || !conservativeValid || !moderateValid || !aggressiveValid) {
      hasInvalidValues = true;
    }
    
    // Format with validation
    const formatSafe = (val) => {
      if (!isFinite(val)) {
        return '<span style="color: var(--color-error); font-weight: bold;">❌ NaN</span>';
      }
      if (val < 0) {
        return '<span style="color: var(--color-warning);">' + formatNumber(0) + '</span>';
      }
      return formatNumber(val);
    };
    
    tr.innerHTML = `
      <td>${state.forecastData.periods[i]}</td>
      <td>${formatSafe(baseline)}</td>
      <td>${formatSafe(conservative)}</td>
      <td>${formatSafe(conservative - baseline)}</td>
      <td>${formatSafe(moderate)}</td>
      <td>${formatSafe(moderate - baseline)}</td>
      <td>${formatSafe(aggressive)}</td>
      <td>${formatSafe(aggressive - baseline)}</td>
    `;
    
    tbody.appendChild(tr);
  }
  
  // Show warning if invalid values detected
  if (hasInvalidValues) {
    const warningDiv = document.createElement('div');
    warningDiv.style.cssText = 'color: var(--color-error); background: rgba(var(--color-error-rgb), 0.1); padding: var(--space-12); border-radius: var(--radius-base); margin-top: var(--space-16);';
    warningDiv.innerHTML = '❌ <strong>警告：</strong>某些預測值無效 (NaN)。請檢查擬合參數和介入參數設定。';
    tbody.parentElement.parentElement.appendChild(warningDiv);
  }
}

function displaySummaryStats() {
  const statsDiv = document.getElementById('summaryStats');
  
  // Add data quality warning if applicable
  if (state.dataQuality && state.dataQuality.score < 80) {
    const warningDiv = document.createElement('div');
    warningDiv.style.cssText = 'grid-column: 1 / -1; background: rgba(var(--color-warning-rgb), 0.1); padding: var(--space-16); border-radius: var(--radius-base); border: 1px solid var(--color-warning); margin-bottom: var(--space-16);';
    warningDiv.innerHTML = `
      <strong>⚠️ 數據品質警告</strong><br>
      <span style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">
        基線數據品質分數為 ${state.dataQuality.score}/100。
        預測不確定性約為 ±${state.forecastData.uncertaintyPercent}%。
        ${state.dataQuality.missingCount > 0 ? '包含 ' + state.dataQuality.missingCount + ' 個插值數據點。' : ''}
        請在報告中標明數據限制。
      </span>
    `;
    statsDiv.appendChild(warningDiv);
  }
  
  // Safe reduction that handles NaN values
  const safeSum = (arr) => {
    return arr.reduce((sum, v) => {
      if (isFinite(v)) {
        return sum + v;
      }
      return sum;
    }, 0);
  };
  
  const safeMax = (arr) => {
    const validValues = arr.filter(v => isFinite(v));
    return validValues.length > 0 ? Math.max(...validValues) : 0;
  };
  
  const totalIncremental = {
    conservative: safeSum(state.forecastData.conservative.map((v, i) => {
      const baseline = state.forecastData.baseline[i];
      if (isFinite(v) && isFinite(baseline)) {
        return v - baseline;
      }
      return 0;
    })),
    moderate: safeSum(state.forecastData.moderate.map((v, i) => {
      const baseline = state.forecastData.baseline[i];
      if (isFinite(v) && isFinite(baseline)) {
        return v - baseline;
      }
      return 0;
    })),
    aggressive: safeSum(state.forecastData.aggressive.map((v, i) => {
      const baseline = state.forecastData.baseline[i];
      if (isFinite(v) && isFinite(baseline)) {
        return v - baseline;
      }
      return 0;
    }))
  };
  
  const peakIncremental = {
    conservative: safeMax(state.forecastData.conservative.map((v, i) => {
      const baseline = state.forecastData.baseline[i];
      if (isFinite(v) && isFinite(baseline)) {
        return v - baseline;
      }
      return 0;
    })),
    moderate: safeMax(state.forecastData.moderate.map((v, i) => {
      const baseline = state.forecastData.baseline[i];
      if (isFinite(v) && isFinite(baseline)) {
        return v - baseline;
      }
      return 0;
    })),
    aggressive: safeMax(state.forecastData.aggressive.map((v, i) => {
      const baseline = state.forecastData.baseline[i];
      if (isFinite(v) && isFinite(baseline)) {
        return v - baseline;
      }
      return 0;
    }))
  };
  
  statsDiv.innerHTML = `
    <div class="summary-card">
      <h4>保守型情境</h4>
      <div class="summary-value">${formatNumber(totalIncremental.conservative)}</div>
      <div class="summary-label">總增量帳戶</div>
      <div class="summary-value" style="font-size: var(--font-size-xl); margin-top: var(--space-8);">${formatNumber(peakIncremental.conservative)}</div>
      <div class="summary-label">峰值季度增量</div>
    </div>
    <div class="summary-card">
      <h4>穩健型情境</h4>
      <div class="summary-value">${formatNumber(totalIncremental.moderate)}</div>
      <div class="summary-label">總增量帳戶</div>
      <div class="summary-value" style="font-size: var(--font-size-xl); margin-top: var(--space-8);">${formatNumber(peakIncremental.moderate)}</div>
      <div class="summary-label">峰值季度增量</div>
    </div>
    <div class="summary-card">
      <h4>積極型情境</h4>
      <div class="summary-value">${formatNumber(totalIncremental.aggressive)}</div>
      <div class="summary-label">總增量帳戶</div>
      <div class="summary-value" style="font-size: var(--font-size-xl); margin-top: var(--space-8);">${formatNumber(peakIncremental.aggressive)}</div>
      <div class="summary-label">峰值季度增量</div>
    </div>
  `;
}

function calculateROI() {
  const devCost = parseFloat(document.getElementById('devCost').value);
  const maintCost = parseFloat(document.getElementById('maintCost').value);
  const acquisitionCost = parseFloat(document.getElementById('acquisitionCost').value);
  const revenuePerAccount = parseFloat(document.getElementById('revenuePerAccount').value);
  
  const forecastYears = parseInt(document.getElementById('forecastYears').value);
  const totalPlatformCost = devCost + (maintCost * forecastYears);
  
  const scenarios = ['conservative', 'moderate', 'aggressive'];
  const roiResults = [];
  
  scenarios.forEach(scenario => {
    const totalIncremental = state.forecastData[scenario].reduce((sum, v, i) => 
      sum + (v - state.forecastData.baseline[i]), 0
    );
    
    const incrementalRevenue = totalIncremental * revenuePerAccount * forecastYears;
    const incrementalAcquisitionCost = totalIncremental * acquisitionCost;
    const netBenefit = incrementalRevenue - incrementalAcquisitionCost - totalPlatformCost;
    const roi = (netBenefit / totalPlatformCost) * 100;
    
    // Calculate payback period (simplified)
    const annualBenefit = netBenefit / forecastYears;
    const paybackYears = annualBenefit > 0 ? totalPlatformCost / (annualBenefit + totalPlatformCost / forecastYears) : Infinity;
    
    roiResults.push({
      scenario,
      incrementalRevenue,
      acquisitionCost: incrementalAcquisitionCost,
      platformCost: totalPlatformCost,
      netBenefit,
      roi,
      payback: paybackYears
    });
  });
  
  displayROIResults(roiResults);
}

function displayROIResults(results) {
  const tbody = document.getElementById('roiTableBody');
  tbody.innerHTML = '';
  
  const scenarioNames = {
    conservative: '保守型',
    moderate: '穩健型',
    aggressive: '積極型'
  };
  
  results.forEach(result => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${scenarioNames[result.scenario]}</td>
      <td>$${formatNumber(result.incrementalRevenue)}</td>
      <td>$${formatNumber(result.acquisitionCost)}</td>
      <td>$${formatNumber(result.platformCost)}</td>
      <td style="font-weight: var(--font-weight-semibold); color: ${result.netBenefit > 0 ? 'var(--color-success)' : 'var(--color-error)'}">$${formatNumber(result.netBenefit)}</td>
      <td style="font-weight: var(--font-weight-semibold);">${result.roi.toFixed(1)}%</td>
      <td>${result.payback < 100 ? result.payback.toFixed(1) : 'N/A'}</td>
    `;
    tbody.appendChild(tr);
  });
  
  document.getElementById('roiResults').style.display = 'block';
}

function exportCSV() {
  let csv = '期間,基線,保守型,保守增量,穩健型,穩健增量,積極型,積極增量\n';
  
  const n = state.forecastData.periods.length;
  for (let i = 0; i < n; i++) {
    const baseline = state.forecastData.baseline[i];
    const conservative = state.forecastData.conservative[i];
    const moderate = state.forecastData.moderate[i];
    const aggressive = state.forecastData.aggressive[i];
    
    const baselineStr = isFinite(baseline) ? baseline.toFixed(0) : 'N/A';
    const conservativeStr = isFinite(conservative) ? conservative.toFixed(0) : 'N/A';
    const moderateStr = isFinite(moderate) ? moderate.toFixed(0) : 'N/A';
    const aggressiveStr = isFinite(aggressive) ? aggressive.toFixed(0) : 'N/A';
    const consIncr = isFinite(conservative) && isFinite(baseline) ? (conservative - baseline).toFixed(0) : 'N/A';
    const modIncr = isFinite(moderate) && isFinite(baseline) ? (moderate - baseline).toFixed(0) : 'N/A';
    const aggIncr = isFinite(aggressive) && isFinite(baseline) ? (aggressive - baseline).toFixed(0) : 'N/A';
    
    csv += `${state.forecastData.periods[i]},${baselineStr},${conservativeStr},${consIncr},${moderateStr},${modIncr},${aggressiveStr},${aggIncr}\n`;
  }
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'forecast_data.csv';
  a.click();
  window.URL.revokeObjectURL(url);
}

function exportChart() {
  const canvas = document.getElementById('forecastChart');
  const url = canvas.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = url;
  a.download = 'forecast_chart.png';
  a.click();
}

// Interpolation methods
function applyInterpolation() {
  const method = document.querySelector('input[name="interpolation"]:checked').value;
  state.interpolationMethod = method;
  
  const data = state.historicalData;
  const missingIndices = data.map((d, i) => d.isMissing ? i : -1).filter(i => i >= 0);
  
  if (missingIndices.length === 0) {
    alert('沒有需要插值的數據');
    return;
  }
  
  if (method === 'ignore') {
    alert('將在擬合時忽略缺漏數據點');
    return;
  }
  
  // Apply interpolation based on method
  missingIndices.forEach(i => {
    let interpolatedValue = null;
    
    if (method === 'linear') {
      interpolatedValue = linearInterpolation(data, i);
    } else if (method === 'gompertz') {
      interpolatedValue = gompertzInterpolation(data, i);
    } else if (method === 'forward') {
      interpolatedValue = forwardFill(data, i);
    }
    
    if (interpolatedValue !== null && isFinite(interpolatedValue)) {
      data[i].accounts = interpolatedValue;
      data[i].isMissing = false;
      data[i].dataType = 'Interpolated';
    }
  });
  
  // Recalculate quality
  analyzeDataQuality();
  displayDataQualityReport();
  displayDataPreview(data);
  
  // Show impact with improvement info
  const oldScore = state.dataQuality ? state.dataQuality.score : 0;
  
  // Recalculate quality after interpolation
  analyzeDataQuality();
  displayDataQualityReport();
  displayDataPreview(data);
  
  const newScore = state.dataQuality.score;
  const improvement = newScore - oldScore;
  
  const impactDiv = document.getElementById('interpolationImpact');
  impactDiv.innerHTML = `
    <h4>✓ 插值完成</h4>
    <p>已使用 <strong>${getMethodName(method)}</strong> 填補 ${missingIndices.length} 個缺漏值。</p>
    <p>數據品質分數: <strong>${oldScore}/100</strong> → <strong style="color: var(--color-success);">${newScore}/100</strong> ${improvement > 0 ? '(+' + improvement.toFixed(0) + ')' : ''}</p>
    <p style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-top: var(--space-8);">
      💡 插值後的值已在數據表中以藍色背景標記。請確認這些值看起來合理。
    </p>
  `;
  impactDiv.style.display = 'block';
  
  // Don't call these again, already called above
  // analyzeDataQuality();
  // displayDataQualityReport();
  // displayDataPreview(data);
}

function getMethodName(method) {
  const names = {
    'linear': '線性插值',
    'gompertz': 'Gompertz 插值',
    'forward': '向前填充'
  };
  return names[method] || method;
}

function linearInterpolation(data, index) {
  // Find previous and next valid values
  let prevIndex = -1, nextIndex = -1;
  
  for (let i = index - 1; i >= 0; i--) {
    if (!data[i].isMissing) {
      prevIndex = i;
      break;
    }
  }
  
  for (let i = index + 1; i < data.length; i++) {
    if (!data[i].isMissing) {
      nextIndex = i;
      break;
    }
  }
  
  if (prevIndex >= 0 && nextIndex >= 0) {
    const prevValue = data[prevIndex].accounts;
    const nextValue = data[nextIndex].accounts;
    const gap = nextIndex - prevIndex;
    const position = index - prevIndex;
    return prevValue + (nextValue - prevValue) * (position / gap);
  } else if (prevIndex >= 0) {
    return data[prevIndex].accounts;
  } else if (nextIndex >= 0) {
    return data[nextIndex].accounts;
  }
  
  return null;
}

function gompertzInterpolation(data, index) {
  // Fit preliminary Gompertz to non-missing data
  const validData = data.filter(d => !d.isMissing).map(d => d.accounts);
  
  if (validData.length < 8) {
    // Fall back to linear if not enough data
    return linearInterpolation(data, index);
  }
  
  const result = fitGompertzCurve(validData);
  if (!result || !result.valid) {
    return linearInterpolation(data, index);
  }
  
  // Evaluate at missing index
  const value = gompertzModel(index, result.params.K, result.params.b, result.params.t0);
  return isFinite(value) ? value : linearInterpolation(data, index);
}

function forwardFill(data, index) {
  // Find previous valid value
  for (let i = index - 1; i >= 0; i--) {
    if (!data[i].isMissing) {
      return data[i].accounts;
    }
  }
  
  // If no previous, find next
  for (let i = index + 1; i < data.length; i++) {
    if (!data[i].isMissing) {
      return data[i].accounts;
    }
  }
  
  return null;
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
  // Load sample data on page load
  loadSampleData();
  
  // Interpolation method change listener
  document.querySelectorAll('input[name="interpolation"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      const method = e.target.value;
      const impactDiv = document.getElementById('interpolationImpact');
      
      const descriptions = {
        'ignore': '跳過缺漏季度。如果缺漏較少（1-2個），這是可行的簡單方法。',
        'linear': '在相鄰值之間繪製直線。適用於趨勢較穩定的時期。',
        'gompertz': '使用 S 型曲線估計。最適合成長數據，通常能提供最準確的估計。',
        'forward': '重複前一個值。保守方法，適合成長停滯期。'
      };
      
      impactDiv.innerHTML = `
        <h4>方法說明</h4>
        <p>${descriptions[method]}</p>
      `;
      impactDiv.style.display = 'block';
    });
  });
  
  // Apply interpolation button
  const applyBtn = document.getElementById('applyInterpolationBtn');
  if (applyBtn) {
    applyBtn.addEventListener('click', applyInterpolation);
  }
  
  // Step 1 events
  document.getElementById('loadSampleBtn').addEventListener('click', loadSampleData);
  document.getElementById('clearDataBtn').addEventListener('click', () => {
    document.getElementById('dataInput').value = '';
    document.getElementById('dataPreviewCard').style.display = 'none';
    state.historicalData = [];
  });
  
  document.getElementById('dataInput').addEventListener('input', parseAndDisplayData);
  document.getElementById('validateDataBtn').addEventListener('click', validateAndProceed);
  
  // Step 2 events
  document.getElementById('backToStep1').addEventListener('click', () => goToStep(1));
  document.getElementById('proceedToStep3').addEventListener('click', () => {
    goToStep(3);
    initializeSliders();
  });
  
  // Step 3 events
  document.querySelectorAll('.scenario-preset').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const scenario = e.target.dataset.scenario;
      loadScenarioPreset(scenario);
    });
  });
  
  document.getElementById('backToStep2').addEventListener('click', () => goToStep(2));
  document.getElementById('proceedToStep4').addEventListener('click', generateForecasts);
  
  // Step 4 events
  document.getElementById('backToStep3').addEventListener('click', () => goToStep(3));
  document.getElementById('resetAllBtn').addEventListener('click', () => {
    if (confirm('確定要重新開始嗎？')) {
      goToStep(1);
      document.getElementById('dataInput').value = '';
      state.historicalData = [];
      state.fittedParams = null;
      state.forecastData = null;
    }
  });
  
  document.getElementById('exportCSVBtn').addEventListener('click', exportCSV);
  document.getElementById('exportChartBtn').addEventListener('click', exportChart);
  document.getElementById('calculateROIBtn').addEventListener('click', calculateROI);
  
  // Initialize sliders
  initializeSliders();
});