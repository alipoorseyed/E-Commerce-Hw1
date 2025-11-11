import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings

warnings.filterwarnings('ignore')

# تنظیمات فارسی برای نمودارها
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

print("=" * 80)
print("شبیه‌سازی رقابت قیمت و تبلیغات در بازار آنلاین")
print("با استفاده از Game Theory و Social Network")
print("=" * 80)

# ================================================================================
# TASK I - آماده‌سازی داده‌ها
# ================================================================================
print("\n[TASK I] آماده‌سازی و پیش‌پردازش داده‌ها")
print("-" * 80)

# خواندن داده‌ها از فایل Excel
file_path = r'D:\DataSet\online_retail_II.xlsx'
print(f"→ در حال خواندن فایل: {file_path}")

try:
    # خواندن هر دو شیت
    print("→ خواندن Sheet 1...")
    df_sheet1 = pd.read_excel(file_path, sheet_name=0)
    print(f"✓ Sheet 1 خوانده شد: {len(df_sheet1):,} رکورد")

    print("→ خواندن Sheet 2...")
    df_sheet2 = pd.read_excel(file_path, sheet_name=1)
    print(f"✓ Sheet 2 خوانده شد: {len(df_sheet2):,} رکورد")

    # ترکیب دو شیت
    print("\n→ در حال ترکیب دو شیت...")
    df = pd.concat([df_sheet1, df_sheet2], ignore_index=True)
    print(f"✓ داده‌ها ترکیب شدند!")
    print(f"  تعداد کل رکوردها: {len(df):,}")
    print(f"  تعداد ستون‌ها: {len(df.columns)}")

    # نمایش نام ستون‌ها
    print(f"\n→ ستون‌های موجود:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    # تطبیق نام ستون‌ها با نام‌های استاندارد
    column_mapping = {}

    for col in df.columns:
        col_lower = col.lower().strip()
        if 'invoice' in col_lower and 'date' not in col_lower:
            column_mapping[col] = 'Invoice'
        elif 'stock' in col_lower or 'code' in col_lower:
            column_mapping[col] = 'StockCode'
        elif 'description' in col_lower or 'desc' in col_lower:
            column_mapping[col] = 'Description'
        elif 'quantity' in col_lower or 'qty' in col_lower:
            column_mapping[col] = 'Quantity'
        elif 'date' in col_lower:
            column_mapping[col] = 'InvoiceDate'
        elif 'price' in col_lower or 'unit' in col_lower:
            column_mapping[col] = 'Price'
        elif 'customer' in col_lower:
            column_mapping[col] = 'CustomerID'
        elif 'country' in col_lower:
            column_mapping[col] = 'Country'

    # تغییر نام ستون‌ها
    df = df.rename(columns=column_mapping)
    print(f"\n→ ستون‌ها استاندارد شدند")

except FileNotFoundError:
    print(f"❌ خطا: فایل در مسیر {file_path} یافت نشد!")
    print("→ استفاده از داده‌های نمونه...")

    np.random.seed(42)
    data = {
        'Invoice': [f'53636{i}' for i in range(1000)],
        'StockCode': np.random.choice(['71053', '84406B', '22086', '21730', '22632'], 1000),
        'Description': np.random.choice(['WHITE METAL LANTERN', 'CREAM CUPID HEARTS',
                                         'PAPER CHAIN KIT'], 1000),
        'Quantity': np.random.randint(1, 50, 1000),
        'InvoiceDate': pd.date_range('2010-12-01', periods=1000, freq='H'),
        'Price': np.random.uniform(0.5, 10.0, 1000).round(2),
        'CustomerID': np.random.randint(12000, 18000, 1000),
        'Country': np.random.choice(['United Kingdom', 'France', 'Germany'], 1000)
    }
    df = pd.DataFrame(data)
except Exception as e:
    print(f"❌ خطا در خواندن فایل: {str(e)}")
    print("→ استفاده از داده‌های نمونه...")

    np.random.seed(42)
    data = {
        'Invoice': [f'53636{i}' for i in range(1000)],
        'StockCode': np.random.choice(['71053', '84406B', '22086', '21730', '22632'], 1000),
        'Description': np.random.choice(['WHITE METAL LANTERN', 'CREAM CUPID HEARTS',
                                         'PAPER CHAIN KIT'], 1000),
        'Quantity': np.random.randint(1, 50, 1000),
        'InvoiceDate': pd.date_range('2010-12-01', periods=1000, freq='H'),
        'Price': np.random.uniform(0.5, 10.0, 1000).round(2),
        'CustomerID': np.random.randint(12000, 18000, 1000),
        'Country': np.random.choice(['United Kingdom', 'France', 'Germany'], 1000)
    }
    df = pd.DataFrame(data)

print(f"\n→ نمونه داده‌ها:")
print(df.head(10))

print(f"\n→ اطلاعات ستون‌ها:")
print(df.info())

# پیش‌پردازش
print("\n→ پیش‌پردازش داده‌ها...")
initial_size = len(df)
print(f"  تعداد رکورد اولیه: {initial_size:,}")

# حذف مقادیر null
df = df.dropna()
print(f"  بعد از حذف null: {len(df):,}")

# حذف مقادیر منفی و نامعتبر
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
print(f"  بعد از حذف مقادیر نامعتبر: {len(df):,}")

# حذف سطرهای تکراری (نه محصولات مشابه)
df = df.drop_duplicates()
print(f"  بعد از حذف تکراری‌ها: {len(df):,}")

removed = initial_size - len(df)
removed_pct = (removed / initial_size) * 100
print(f"\n✓ پاکسازی کامل شد! حذف شده: {removed:,} رکورد ({removed_pct:.1f}%)")

# بررسی اجمالی
print("\n→ آمار توصیفی داده‌های عددی:")
print(df[['Quantity', 'Price']].describe())

print("\n→ آمار توصیفی هر محصول (Top 10):")
product_stats = df.groupby('Description').agg({
    'Price': ['mean', 'min', 'max'],
    'Quantity': ['sum', 'mean', 'count']
}).round(2)
print(product_stats.head(10))

# ================================================================================
# TASK II - محاسبه تابع سود برای دو فروشنده
# ================================================================================
print("\n\n[TASK II] محاسبه تابع سود برای دو فروشنده")
print("-" * 80)

# تعریف پارامترهای ثابت
print("\n→ تعریف پارامترهای ثابت بازی:")
COST = 2.0  # هزینه تولید
BASE_DEMAND = 100  # تقاضای پایه
ALPHA = 2.0  # تأثیر تبلیغات
BETA = -15.0  # حساسیت به اختلاف قیمت
GAMMA = 0.5  # تأثیر شبکه اجتماعی (در Task II صفر است)
INFLUENCE_SCORE = 0  # در Task II صفر

print(f"  هزینه تولید (cost): {COST}")
print(f"  تقاضای پایه (base_demand): {BASE_DEMAND}")
print(f"  ضریب تبلیغات (α): {ALPHA}")
print(f"  ضریب قیمت (β): {BETA}")
print(f"  ضریب تأثیر اجتماعی (γ): {GAMMA}")
print(f"  Influence Score: {INFLUENCE_SCORE}")

# تعریف محدوده قیمت و بودجه تبلیغات
print("\n→ تعریف استراتژی‌های ممکن:")
price_options = np.linspace(3.0, 8.0, 10)  # 10 قیمت مختلف
ad_options = np.linspace(5.0, 35.0, 6)  # 6 بودجه تبلیغات

print(f"  قیمت‌های ممکن (10 حالت): {[f'{p:.1f}' for p in price_options]}")
print(f"  بودجه تبلیغات (6 حالت): {[f'{a:.1f}' for a in ad_options]}")
print(f"  تعداد کل استراتژی‌ها: {len(price_options)} × {len(ad_options)} = {len(price_options) * len(ad_options)}")


# تابع محاسبه تقاضا
def calculate_demand(price_i, price_j, ad_budget_i, influence=0):
    """
    محاسبه تقاضا برای فروشنده i
    D_i = base_demand + α×m_i + β×(p_i - p_j) + γ×influence
    """
    demand = BASE_DEMAND + (ALPHA * ad_budget_i) + (BETA * (price_i - price_j)) + (GAMMA * influence)
    return max(0, demand)


# تابع محاسبه سود
def calculate_profit(price_i, price_j, ad_budget_i, influence=0):
    """
    محاسبه سود برای فروشنده i
    Profit_i = (p_i - cost) × D_i - m_i
    """
    demand = calculate_demand(price_i, price_j, ad_budget_i, influence)
    profit = (price_i - COST) * demand - ad_budget_i
    return profit


# ایجاد ماتریس سود برای هر فروشنده
print("\n→ محاسبه ماتریس سود برای دو فروشنده...")

n_strategies = len(price_options) * len(ad_options)
print(f"  اندازه ماتریس: {n_strategies} × {n_strategies}")

# لیست تمام استراتژی‌ها (قیمت, بودجه تبلیغات)
strategies = [(p, a) for p in price_options for a in ad_options]

# ماتریس سود فروشنده 1
profit_matrix_seller1 = np.zeros((n_strategies, n_strategies))

# ماتریس سود فروشنده 2
profit_matrix_seller2 = np.zeros((n_strategies, n_strategies))

print("  در حال محاسبه...")
for i, (p1, a1) in enumerate(strategies):
    for j, (p2, a2) in enumerate(strategies):
        # سود فروشنده 1 با استراتژی i در برابر استراتژی j فروشنده 2
        profit_matrix_seller1[i, j] = calculate_profit(p1, p2, a1, INFLUENCE_SCORE)

        # سود فروشنده 2 با استراتژی j در برابر استراتژی i فروشنده 1
        profit_matrix_seller2[j, i] = calculate_profit(p2, p1, a2, INFLUENCE_SCORE)

print("✓ ماتریس‌های سود محاسبه شدند")

# نمایش نمونه از ماتریس سود
print("\n→ نمونه از ماتریس سود فروشنده 1 (5×5 اول):")
print(profit_matrix_seller1[:5, :5].round(2))

# ================================================================================
# TASK III - پیدا کردن تعادل نش
# ================================================================================
print("\n\n[TASK III] پیدا کردن تعادل نش")
print("-" * 80)

print("\n→ جستجوی تعادل نش در ماتریس‌های سود...")
print("  تعادل نش: استراتژی‌هایی که هیچ بازیکنی انگیزه‌ای برای تغییر ندارد")

nash_equilibria = []

for i in range(n_strategies):
    for j in range(n_strategies):
        # بررسی آیا (i, j) یک تعادل نش است

        # بهترین پاسخ فروشنده 1 به استراتژی j فروشنده 2
        best_response_s1 = np.argmax(profit_matrix_seller1[:, j])

        # بهترین پاسخ فروشنده 2 به استراتژی i فروشنده 1
        best_response_s2 = np.argmax(profit_matrix_seller2[:, i])

        # اگر (i, j) بهترین پاسخ متقابل باشد، تعادل نش است
        if best_response_s1 == i and best_response_s2 == j:
            nash_equilibria.append((i, j))

print(f"\n✓ تعداد تعادل‌های نش پیدا شده: {len(nash_equilibria)}")

if len(nash_equilibria) > 0:
    print("\n→ تعادل‌های نش:")
    print(f"{'#':<4} {'S1_Strategy':<15} {'S1_Price':<10} {'S1_Ad':<10} {'S1_Profit':<12} "
          f"{'S2_Strategy':<15} {'S2_Price':<10} {'S2_Ad':<10} {'S2_Profit':<12}")
    print("-" * 120)

    for idx, (i, j) in enumerate(nash_equilibria, 1):
        p1, a1 = strategies[i]
        p2, a2 = strategies[j]
        profit1 = profit_matrix_seller1[i, j]
        profit2 = profit_matrix_seller2[j, i]

        print(f"{idx:<4} {f'({i})':<15} {p1:<10.2f} {a1:<10.2f} {profit1:<12.2f} "
              f"{f'({j})':<15} {p2:<10.2f} {a2:<10.2f} {profit2:<12.2f}")

    # انتخاب اولین تعادل نش برای تحلیل بیشتر
    nash_i, nash_j = nash_equilibria[0]
    nash_p1, nash_a1 = strategies[nash_i]
    nash_p2, nash_a2 = strategies[nash_j]
    nash_profit1 = profit_matrix_seller1[nash_i, nash_j]
    nash_profit2 = profit_matrix_seller2[nash_j, nash_i]

    print(f"\n→ تعادل نش اصلی برای تصویرسازی:")
    print(f"  Seller 1: Price={nash_p1:.2f}, Ad={nash_a1:.2f}, Profit={nash_profit1:.2f}")
    print(f"  Seller 2: Price={nash_p2:.2f}, Ad={nash_a2:.2f}, Profit={nash_profit2:.2f}")
else:
    print("\n⚠ هیچ تعادل نش خالصی پیدا نشد!")
    # استفاده از استراتژی میانگین
    nash_i = n_strategies // 2
    nash_j = n_strategies // 2
    nash_p1, nash_a1 = strategies[nash_i]
    nash_p2, nash_a2 = strategies[nash_j]
    nash_profit1 = profit_matrix_seller1[nash_i, nash_j]
    nash_profit2 = profit_matrix_seller2[nash_j, nash_i]

# ================================================================================
# TASK IV - محاسبه Influence Score از شبکه مشتریان
# ================================================================================
print("\n\n[TASK IV] محاسبه Influence Score از شبکه اجتماعی مشتریان")
print("-" * 80)

print("\n→ ایجاد گراف شبکه مشتریان بر اساس محصولات مشترک...")

# فیلتر کردن مشتریان با CustomerID معتبر
df_customers = df[df['CustomerID'].notna()].copy()
print(f"  تعداد تراکنش‌ها با CustomerID معتبر: {len(df_customers):,}")

# ایجاد لیست محصولات خریداری شده توسط هر مشتری
customer_products = df_customers.groupby('CustomerID')['StockCode'].apply(set).to_dict()
print(f"  تعداد مشتریان منحصر به فرد: {len(customer_products):,}")

# ایجاد گراف
print("\n→ ساخت گراف...")
G = nx.Graph()

# افزودن مشتریان به عنوان نود
customers = list(customer_products.keys())
G.add_nodes_from(customers)

print(f"  تعداد نودها (مشتریان): {G.number_of_nodes():,}")

# افزودن یال‌ها: دو مشتری متصل هستند اگر حداقل یک محصول مشترک خریده باشند
print("  در حال اضافه کردن یال‌ها...")
edge_count = 0

for i, customer_i in enumerate(customers):
    if i % 1000 == 0:
        print(f"    پردازش شده: {i:,} / {len(customers):,} مشتری")

    products_i = customer_products[customer_i]

    # فقط با مشتریان بعدی مقایسه کنیم (برای جلوگیری از تکرار)
    for customer_j in customers[i + 1:]:
        products_j = customer_products[customer_j]

        # اگر محصول مشترک دارند، یال اضافه کن
        if len(products_i & products_j) > 0:
            G.add_edge(customer_i, customer_j)
            edge_count += 1

print(f"✓ گراف ساخته شد!")
print(f"  تعداد یال‌ها (ارتباطات): {G.number_of_edges():,}")
print(f"  میانگین درجه نودها: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")

# محاسبه Influence Score
print("\n→ محاسبه Influence Score...")

# درجه هر نود
degrees = dict(G.degree())

# میانگین درجه‌ها
avg_degree = np.mean(list(degrees.values()))

# Influence Score = میانگین درجه / تعداد مشتریان
influence_score = avg_degree / G.number_of_nodes()

print(f"  میانگین درجه نودها: {avg_degree:.4f}")
print(f"  تعداد مشتریان: {G.number_of_nodes()}")
print(f"✓ Influence Score محاسبه شد: {influence_score:.6f}")

# محاسبه مجدد سود با Influence Score
print("\n→ محاسبه مجدد سود با Influence Score...")

profit_with_influence_s1 = calculate_profit(nash_p1, nash_p2, nash_a1, influence_score)
profit_with_influence_s2 = calculate_profit(nash_p2, nash_p1, nash_a2, influence_score)

print(f"\n→ مقایسه سود (بدون vs با تأثیر اجتماعی):")
print(f"\n  Seller 1:")
print(f"    بدون تأثیر اجتماعی: {nash_profit1:.2f}")
print(f"    با تأثیر اجتماعی: {profit_with_influence_s1:.2f}")
print(
    f"    افزایش: {profit_with_influence_s1 - nash_profit1:.2f} ({((profit_with_influence_s1 - nash_profit1) / nash_profit1 * 100):.2f}%)")

print(f"\n  Seller 2:")
print(f"    بدون تأثیر اجتماعی: {nash_profit2:.2f}")
print(f"    با تأثیر اجتماعی: {profit_with_influence_s2:.2f}")
print(
    f"    افزایش: {profit_with_influence_s2 - nash_profit2:.2f} ({((profit_with_influence_s2 - nash_profit2) / nash_profit2 * 100):.2f}%)")

# ================================================================================
# TASK V - تصویرسازی
# ================================================================================
print("\n\n[TASK V] تصویرسازی نتایج")
print("-" * 80)

color1, color2 = '#FF6B6B', '#4ECDC4'

# نمودار 1: ماتریس سود برای هر فروشنده (Heatmap)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

im1 = axes[0].imshow(profit_matrix_seller1, cmap='RdYlGn', aspect='auto')
axes[0].set_title('Profit Matrix - Seller 1', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Seller 2 Strategy Index', fontsize=11)
axes[0].set_ylabel('Seller 1 Strategy Index', fontsize=11)
plt.colorbar(im1, ax=axes[0], label='Profit')

if len(nash_equilibria) > 0:
    for nash_i, nash_j in nash_equilibria:
        axes[0].scatter(nash_j, nash_i, color='blue', s=200, marker='*',
                        edgecolor='white', linewidth=2, zorder=5)

im2 = axes[1].imshow(profit_matrix_seller2, cmap='RdYlGn', aspect='auto')
axes[1].set_title('Profit Matrix - Seller 2', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Seller 1 Strategy Index', fontsize=11)
axes[1].set_ylabel('Seller 2 Strategy Index', fontsize=11)
plt.colorbar(im2, ax=axes[1], label='Profit')

if len(nash_equilibria) > 0:
    for nash_i, nash_j in nash_equilibria:
        axes[1].scatter(nash_i, nash_j, color='blue', s=200, marker='*',
                        edgecolor='white', linewidth=2, zorder=5)

plt.tight_layout()
plt.savefig('profit_matrices.png', dpi=300, bbox_inches='tight')
print("✓ نمودار ماتریس سود ذخیره شد: profit_matrices.png")

# نمودار 2: تأثیر قیمت و تبلیغات بر سود (3D Surface)
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
P, A = np.meshgrid(price_options, ad_options)
Z = np.zeros_like(P)

for i in range(len(ad_options)):
    for j in range(len(price_options)):
        Z[i, j] = calculate_profit(P[i, j], nash_p2, A[i, j], INFLUENCE_SCORE)

surf1 = ax1.plot_surface(P, A, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('Price', fontsize=10)
ax1.set_ylabel('Ad Budget', fontsize=10)
ax1.set_zlabel('Profit', fontsize=10)
ax1.set_title('Seller 1 Profit Landscape\n(vs Seller 2 Nash Strategy)', fontsize=12, fontweight='bold')
plt.colorbar(surf1, ax=ax1, shrink=0.5)

ax2 = fig.add_subplot(122, projection='3d')
Z2 = np.zeros_like(P)

for i in range(len(ad_options)):
    for j in range(len(price_options)):
        Z2[i, j] = calculate_profit(P[i, j], nash_p1, A[i, j], INFLUENCE_SCORE)

surf2 = ax2.plot_surface(P, A, Z2, cmap='plasma', alpha=0.8)
ax2.set_xlabel('Price', fontsize=10)
ax2.set_ylabel('Ad Budget', fontsize=10)
ax2.set_zlabel('Profit', fontsize=10)
ax2.set_title('Seller 2 Profit Landscape\n(vs Seller 1 Nash Strategy)', fontsize=12, fontweight='bold')
plt.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.savefig('profit_3d_landscape.png', dpi=300, bbox_inches='tight')
print("✓ نمودار 3D سود ذخیره شد: profit_3d_landscape.png")

# نمودار 3: تأثیر شبکه اجتماعی
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# مقایسه سود
comparison = pd.DataFrame({
    'Without Social Network': [nash_profit1, nash_profit2],
    'With Social Network': [profit_with_influence_s1, profit_with_influence_s2]
}, index=['Seller 1', 'Seller 2'])

comparison.plot(kind='bar', ax=axes[0], color=[color1, color2], alpha=0.8)
axes[0].set_title('Impact of Social Network on Profit', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Profit', fontsize=11)
axes[0].set_xlabel('Seller', fontsize=11)
axes[0].legend(title='Scenario', fontsize=9)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# نمایش گراف شبکه مشتریان (نمونه کوچک)
print("\n→ آماده‌سازی نمایش گراف شبکه...")
# برای نمایش، یک زیرگراف کوچک انتخاب می‌کنیم
if G.number_of_nodes() > 100:
    # انتخاب 100 نود با بیشترین درجه
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:100]
    top_node_ids = [node for node, _ in top_nodes]
    G_sample = G.subgraph(top_node_ids).copy()
else:
    G_sample = G

pos = nx.spring_layout(G_sample, seed=42, k=0.5)
node_sizes = [degrees[n] * 20 for n in G_sample.nodes()]
node_colors = [degrees[n] for n in G_sample.nodes()]

nx.draw(G_sample, pos, ax=axes[1], node_size=node_sizes, node_color=node_colors,
        cmap='YlOrRd', with_labels=False, edge_color='gray', alpha=0.6, width=0.5)
axes[1].set_title(f'Customer Social Network\n({G_sample.number_of_nodes()} customers, '
                  f'Influence Score={influence_score:.6f})',
                  fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('social_network_impact.png', dpi=300, bbox_inches='tight')
print("✓ نمودار تأثیر شبکه اجتماعی ذخیره شد: social_network_impact.png")

# نمودار 4: توزیع درجات در گراف
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

degree_values = list(degrees.values())
ax.hist(degree_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(avg_degree, color='red', linestyle='--', linewidth=2, label=f'Mean Degree = {avg_degree:.2f}')
ax.set_xlabel('Degree (Number of Connections)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Degree Distribution in Customer Network', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('degree_distribution.png', dpi=300, bbox_inches='tight')
print("✓ نمودار توزیع درجات ذخیره شد: degree_distribution.png")

# نمودار 5: مقایسه استراتژی‌های مختلف
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. تأثیر قیمت بر سود (با بودجه تبلیغات ثابت)
fixed_ad = ad_options[len(ad_options) // 2]
profits_s1 = [calculate_profit(p, nash_p2, fixed_ad, INFLUENCE_SCORE) for p in price_options]
profits_s2 = [calculate_profit(p, nash_p1, fixed_ad, INFLUENCE_SCORE) for p in price_options]

axes[0, 0].plot(price_options, profits_s1, marker='o', color=color1, linewidth=2, label='Seller 1')
axes[0, 0].plot(price_options, profits_s2, marker='s', color=color2, linewidth=2, label='Seller 2')
axes[0, 0].axvline(nash_p1, color=color1, linestyle='--', alpha=0.5)
axes[0, 0].axvline(nash_p2, color=color2, linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Price', fontsize=11)
axes[0, 0].set_ylabel('Profit', fontsize=11)
axes[0, 0].set_title(f'Price Impact on Profit (Ad Budget = {fixed_ad:.1f})', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. تأثیر بودجه تبلیغات بر سود (با قیمت ثابت)
fixed_price = price_options[len(price_options) // 2]
profits_ad_s1 = [calculate_profit(fixed_price, nash_p2, a, INFLUENCE_SCORE) for a in ad_options]
profits_ad_s2 = [calculate_profit(fixed_price, nash_p1, a, INFLUENCE_SCORE) for a in ad_options]

axes[0, 1].plot(ad_options, profits_ad_s1, marker='o', color=color1, linewidth=2, label='Seller 1')
axes[0, 1].plot(ad_options, profits_ad_s2, marker='s', color=color2, linewidth=2, label='Seller 2')
axes[0, 1].axvline(nash_a1, color=color1, linestyle='--', alpha=0.5)
axes[0, 1].axvline(nash_a2, color=color2, linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Advertising Budget', fontsize=11)
axes[0, 1].set_ylabel('Profit', fontsize=11)
axes[0, 1].set_title(f'Advertising Impact on Profit (Price = {fixed_price:.1f})', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. تقاضا در تعادل نش
demand_nash_s1 = calculate_demand(nash_p1, nash_p2, nash_a1, INFLUENCE_SCORE)
demand_nash_s2 = calculate_demand(nash_p2, nash_p1, nash_a2, INFLUENCE_SCORE)
demand_with_inf_s1 = calculate_demand(nash_p1, nash_p2, nash_a1, influence_score)
demand_with_inf_s2 = calculate_demand(nash_p2, nash_p1, nash_a2, influence_score)

demand_comparison = pd.DataFrame({
    'Without Social': [demand_nash_s1, demand_nash_s2],
    'With Social': [demand_with_inf_s1, demand_with_inf_s2]
}, index=['Seller 1', 'Seller 2'])

demand_comparison.plot(kind='bar', ax=axes[1, 0], color=[color1, color2], alpha=0.8)
axes[1, 0].set_title('Demand Comparison at Nash Equilibrium', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Demand', fontsize=11)
axes[1, 0].set_xlabel('Seller', fontsize=11)
axes[1, 0].legend(title='Scenario', fontsize=9)
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

# 4. نمودار استراتژی Nash Equilibrium
if len(nash_equilibria) > 0:
    nash_prices_s1 = [strategies[i][0] for i, j in nash_equilibria]
    nash_ads_s1 = [strategies[i][1] for i, j in nash_equilibria]
    nash_prices_s2 = [strategies[j][0] for i, j in nash_equilibria]
    nash_ads_s2 = [strategies[j][1] for i, j in nash_equilibria]

    axes[1, 1].scatter(nash_prices_s1, nash_ads_s1, s=150, color=color1,
                       marker='*', edgecolor='black', linewidth=1.5, label='Seller 1 Nash', zorder=5)
    axes[1, 1].scatter(nash_prices_s2, nash_ads_s2, s=150, color=color2,
                       marker='*', edgecolor='black', linewidth=1.5, label='Seller 2 Nash', zorder=5)

    # نمایش تمام استراتژی‌های ممکن
    all_prices = [s[0] for s in strategies]
    all_ads = [s[1] for s in strategies]
    axes[1, 1].scatter(all_prices, all_ads, s=20, color='lightgray', alpha=0.3, label='All Strategies')

    axes[1, 1].set_xlabel('Price', fontsize=11)
    axes[1, 1].set_ylabel('Advertising Budget', fontsize=11)
    axes[1, 1].set_title(f'Nash Equilibria in Strategy Space ({len(nash_equilibria)} found)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, 'No Pure Nash Equilibrium Found',
                    ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Nash Equilibrium Search Result', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('strategy_analysis.png', dpi=300, bbox_inches='tight')
print("✓ نمودار تحلیل استراتژی ذخیره شد: strategy_analysis.png")

print("\n" + "=" * 80)
print("✓ شبیه‌سازی با موفقیت تکمیل شد!")
print("=" * 80)

# خلاصه نتایج نهایی
print("\n" + "=" * 80)
print("خلاصه نتایج")
print("=" * 80)

print(f"\n[1] پارامترهای بازی:")
print(f"    • هزینه تولید: {COST}")
print(f"    • تقاضای پایه: {BASE_DEMAND}")
print(f"    • ضریب تبلیغات (α): {ALPHA}")
print(f"    • ضریب قیمت (β): {BETA}")
print(f"    • ضریب شبکه اجتماعی (γ): {GAMMA}")

print(f"\n[2] فضای استراتژی:")
print(f"    • تعداد قیمت‌های ممکن: {len(price_options)}")
print(f"    • تعداد بودجه‌های تبلیغاتی: {len(ad_options)}")
print(f"    • تعداد کل استراتژی‌ها: {n_strategies}")

print(f"\n[3] تعادل‌های نش:")
print(f"    • تعداد تعادل‌های نش یافت شده: {len(nash_equilibria)}")
if len(nash_equilibria) > 0:
    print(f"    • تعادل نش اصلی:")
    print(f"      - Seller 1: Price={nash_p1:.2f}, Ad={nash_a1:.2f}, Profit={nash_profit1:.2f}")
    print(f"      - Seller 2: Price={nash_p2:.2f}, Ad={nash_a2:.2f}, Profit={nash_profit2:.2f}")

print(f"\n[4] شبکه اجتماعی:")
print(f"    • تعداد مشتریان: {G.number_of_nodes():,}")
print(f"    • تعداد ارتباطات: {G.number_of_edges():,}")
print(f"    • میانگین درجه: {avg_degree:.2f}")
print(f"    • Influence Score: {influence_score:.6f}")

print(f"\n[5] تأثیر شبکه اجتماعی:")
print(f"    • افزایش سود Seller 1: {profit_with_influence_s1 - nash_profit1:.2f} "
      f"({((profit_with_influence_s1 - nash_profit1) / nash_profit1 * 100):.2f}%)")
print(f"    • افزایش سود Seller 2: {profit_with_influence_s2 - nash_profit2:.2f} "
      f"({((profit_with_influence_s2 - nash_profit2) / nash_profit2 * 100):.2f}%)")

print(f"\n[6] فایل‌های خروجی:")
print(f"    ✓ profit_matrices.png")
print(f"    ✓ profit_3d_landscape.png")
print(f"    ✓ social_network_impact.png")
print(f"    ✓ degree_distribution.png")
print(f"    ✓ strategy_analysis.png")

print("\n" + "=" * 80)
