# HAR Recording Task List

The browser agent performs all of these tasks in a single session with network recording enabled. Every request is captured — no filtering needed. One HAR dump is exported at the end.

**Credentials used throughout:**
- Shopping (customer): `emma.lopez@gmail.com` / `Password.1`
- Shopping Admin: `admin` / `admin1234`
- Forum: `MarvelsGrantMan136` / `test1234`

---

## App 1 — Shopping (port 7770)
`http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/`

### Guest flows (no login)

1. Open the homepage.
2. Click into the **Beauty & Personal Care** top-level category from the nav.
3. Navigate to **Beauty & Personal Care > Oral Care > Toothbrushes & Accessories** — let the product list load.
4. Search for `"ginger"` using the search bar — let results load.
5. Click on any product from the search results — let the product detail page load fully.
6. Add that product to the cart (select a quantity if required, click Add to Cart).
7. Click the cart icon — open the mini-cart.
8. Click **Proceed to Checkout**.
9. Fill in the guest checkout shipping form:
   - Email: `test@example.com`
   - First Name: `Test`
   - Last Name: `User`
   - Street: `123 Main St`
   - City: `New York`
   - State: `New York`
   - ZIP: `10001`
   - Country: `United States`
   - Phone: `5551234567`
10. Select the first available shipping method and click **Next**.
11. On the payment step, leave the default payment method and click **Place Order**.

### Logged-in customer flows

12. Log in with `emma.lopez@gmail.com` / `Password.1` (My Account → Sign In).
13. After login, open **My Account** dashboard.
14. Navigate to **My Orders** under the account sidebar.
15. Click into any existing order to view its detail page.
16. Go to **My Wishlist** (account sidebar).
17. Navigate to a product — **Sports & Outdoors > Exercise & Fitness** — pick any product.
18. Click **Add to Wish List** on that product.
19. Go to **My Wishlist** again to confirm it was added.
20. From the wishlist, click **Add to Cart** for that same product.
21. Go to the cart, change the quantity of one item to `2`, and click **Update Cart**.
22. Navigate to **My Account > Address Book** and view existing addresses.
23. Log out.

---

## App 2 — Shopping Admin (port 7780)
`http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7780/admin`

> **Note:** Port 7780 serves the customer-facing storefront at the root URL. The Magento Admin panel is at the `/admin` subpath. The browser agent must navigate directly to `/admin` to reach the admin login page.

### Authentication

24. Go to the admin login page at `http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7780/admin`.
25. Log in with `admin` / `admin1234`.

### Catalog management

26. Navigate to **Catalog > Products** from the left sidebar.
27. Let the product grid load with the default filters.
28. Use the search/filter bar to filter products by **Name** containing `"tee"` — apply filters.
29. Click into any product from the filtered list to open the product edit form.
30. Change the product **Price** to any nearby value (e.g., add $1), scroll down to **Save**.
31. Navigate back to **Catalog > Products**.
32. Click **Add Product** (top right) — select **Simple Product** if prompted.
33. Fill in the new product form:
    - Product Name: `HAR Test Product`
    - SKU: `HAR-TEST-001`
    - Price: `19.99`
    - Quantity: `100`
    - Attribute Set: Default
34. Click **Save** on the new product.

### Order management

35. Navigate to **Sales > Orders** from the left sidebar.
36. Let the order grid load.
37. Click into any existing order to open the order detail view.
38. Note the order status. Click **Invoice** (if the button is available) — fill in the invoice form defaults and click **Submit Invoice**.

### Customer management

39. Navigate to **Customers > All Customers**.
40. Click into any customer record to view the account detail page.
41. In the customer account page, click the **Orders** tab to see their order history.

### Reports

42. Navigate to **Reports > Products > Bestsellers**.
43. Navigate to **Reports > Sales > Orders** — let the report load.

### Logout

44. Log out from the admin panel (Admin menu, top right → Sign Out).

---

## App 3 — Forum (port 9999)
`http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:9999/`

### Guest browsing

45. Open the forum homepage.
46. Click on **Forums** in the nav — let the forum list load.
47. Click into any available forum/subforum.
48. Click on any post/thread to open it.
49. Click on any user's username to view their profile page.

### Authenticated flows

50. Click **Log In** and sign in with `MarvelsGrantMan136` / `test1234`.
51. After login, return to the homepage — confirm you are logged in.
52. Click into a forum that allows posting.
53. Click **New Thread** or **Submit Link** or **Submit Text** (whatever button is present for creating a post).
54. Fill in the post form:
    - Title: `HAR Test Post - API Coverage`
    - Body/URL: `This is a test post created for HAR recording.`
55. Submit the post.
56. After submitting, view the created post's page.
57. On the post, click the **Comment** or reply area — type a comment: `"Test comment for HAR recording."` — submit it.
58. On any other post (not your own), click the **upvote** button.
59. On any post, click **Save** / bookmark if the option exists.
60. Navigate to your own profile page (click your username in the top bar).
61. Click **Log Out**.

---

## App 4 — Map (port 3000)
`http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:3000/`

### Browse and search

62. Open the map homepage — let the default map tiles load.
63. In the **Search** bar at the top, type `"New York"` and press enter/click search — let results load and map pan.
64. Click on one of the search results to zoom into that location.
65. Zoom in several levels using the `+` button or scroll wheel.
66. Zoom back out using the `−` button.
67. Pan the map by clicking and dragging to a different area.
68. Search for `"London"` — let results load.
69. Click **Export** in the top nav — let the export panel open (you don't need to actually download).
70. Click on the map to drop a marker — then click **Where is this?** in the top bar with the marker active.

### Node/way detail

71. In the search box, search for `"Central Park"` — click the result.
72. Click on any map feature (node/way) that becomes clickable — let the sidebar panel load the feature detail.

---

## Coverage cross-check

After completing all tasks above, you should have HAR traffic covering:

| App | Auth endpoints | Product/content listing | Item creation/mutation | Session/cookie flows |
|-----|---------------|------------------------|----------------------|---------------------|
| Shopping (guest) | — | ✓ category, search | ✓ cart, checkout | ✓ guest session |
| Shopping (authed) | ✓ login, logout | ✓ orders, wishlist | ✓ wishlist add, cart update | ✓ customer token |
| Admin | ✓ admin login/logout | ✓ product grid, order grid | ✓ product edit, create, invoice | ✓ admin token |
| Forum (guest) | — | ✓ forums, posts | — | — |
| Forum (authed) | ✓ login, logout | ✓ profile | ✓ post create, comment, vote | ✓ CSRF form_key |
| Map | — | ✓ tile loads, search | — | — |

---

## Initial Run — Browser Agent Tasks for the 7 Training Templates

The full task list above covers broad application exploration. For the initial training run, the browser agent only needs to complete the tasks that produce HAR traffic relevant to the **7 task templates defined in [README.md](README.md)**. Below is the minimum set grouped by application so the browser agent can work through one app at a time in a single session.

---

### Shopping (port 7770) — Templates 1, 3, 6
`http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/`

Covers: category listing (Easy), add-to-cart (Medium), and full guest checkout (Hard).

The browser agent runs the guest checkout flow end-to-end. This single pass captures all the HAR traffic needed for all three Shopping templates — category browsing produces Template 1 traffic, cart creation + item addition produces Template 3 traffic, and the full checkout completes Template 6.

1. Open the Shopping homepage.
2. Click into **Beauty & Personal Care** from the nav. *(Template 1: category listing)*
3. Navigate to **Beauty & Personal Care > Oral Care > Toothbrushes & Accessories** — let the product list load. *(Template 1: product listing)*
4. Search for `"ginger"` using the search bar — let results load. *(Template 3: product lookup)*
5. Click on any product from the search results — let the product detail page load fully. *(Template 3: product detail)*
6. Add that product to the cart. *(Template 3: cart creation + item addition)*
7. Click the cart icon — open the mini-cart. *(Template 3: cart state)*
8. Click **Proceed to Checkout**. *(Template 6: checkout begins)*
9. Fill in the guest checkout shipping form: *(Template 6: shipping)*
   - Email: `test@example.com`
   - First Name: `Test`, Last Name: `User`
   - Street: `123 Main St`, City: `New York`, State: `New York`, ZIP: `10001`
   - Country: `United States`, Phone: `5551234567`
10. Select the first available shipping method and click **Next**. *(Template 6: shipping method)*
11. On the payment step, leave the default payment method and click **Place Order**. *(Template 6: payment + order)*

**HAR traffic captured:** category tree API, product list/search API, guest cart creation, add-to-cart, estimate-shipping, set-shipping-information, payment-information, place-order.

---

### Shopping Admin (port 7780) — Template 7
`http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7780/admin`

Covers: admin product creation (Hard).

> **Note:** The root URL on port 7780 shows the customer storefront, not the admin panel. The browser agent must navigate to `/admin` to reach the admin login.

1. Go to the admin login page at `http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7780/admin`.
2. Log in with `admin` / `admin1234`.
3. Click **Add Product** (top right) — select **Simple Product** if prompted.
4. Fill in the new product form:
   - Product Name: `HAR Test Product`
   - SKU: `HAR-TEST-001`
   - Price: `19.99`
   - Quantity: `100`
   - Attribute Set: Default
5. Click **Save** on the new product.

**HAR traffic captured:** admin auth token flow, product creation POST with full Magento product schema.

---

### Forum (port 9999) — Templates 4, 5
`http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:9999/`

Covers: authenticated category browsing (Medium) and post creation (Hard).

The browser agent logs in once, browses categories, then creates a post. This single pass captures traffic for both Forum templates — browsing produces Template 4 traffic, and post creation produces Template 5 traffic.

1. Open the Forum homepage.
2. Click on **Forums** in the nav — let the forum list load. *(Template 4: category listing)*
3. Log in with `MarvelsGrantMan136` / `test1234`. *(Templates 4 & 5: auth + CSRF token)*
4. After login, return to the homepage — confirm logged in.
5. Click into any available forum/subforum. *(Template 4: authed category browse)*
6. Click on any post/thread to open it. *(Template 4: authed post retrieval)*
7. Navigate to a forum that allows posting. *(Template 5: post creation begins)*
8. Click **New Thread** / **Submit Text**. *(Template 5: creation form)*
9. Fill in the post form: *(Template 5: post body)*
   - Title: `HAR Test Post - API Coverage`
   - Body: `This is a test post created for HAR recording.`
10. Submit the post. *(Template 5: POST with CSRF form_key)*
11. View the created post's page. *(Template 5: confirm creation)*

**HAR traffic captured:** login + session/CSRF extraction, forum/subforum listing (authed), thread listing, post creation with form_key.

---

### Wikipedia (port 8888) — Template 2
`http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:8888/`

Covers: article summary retrieval (Easy).

Wikipedia is not covered in the full task list above — these are new tasks for the initial run.

1. Open the Wikipedia homepage.
2. Search for any article (e.g., `"Python (programming language)"`).
3. Click into the article and let the full page load.

**HAR traffic captured:** Kiwix search API, article content retrieval API.

---

### What is NOT needed for the initial run

| Skipped section | Tasks | Why not needed |
|----------------|-------|----------------|
| Shopping — logged-in customer flows | 12–23 | No template targets authed customer actions (orders, wishlist, address book) |
| Admin — catalog editing | 26–31 | Template 7 only needs product *creation*, not editing existing products |
| Admin — orders, customers, reports | 35–44 | No template targets admin read flows |
| Forum — voting, commenting, bookmarking | 57–61 | Templates 4 & 5 cover browse and post creation only |
| Map (port 3000) | 62–72 | No template targets the Map application |
