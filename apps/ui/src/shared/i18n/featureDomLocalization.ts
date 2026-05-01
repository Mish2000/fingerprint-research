import { useEffect } from "react";
import type { AppLanguage } from "../preferences/types.ts";

const HEBREW_TEXT: Record<string, string> = {
    "Verify Workspace": "סביבת אימות",
    "Run curated demos, browse real datasets, or upload your own pair.": "הרצת דמואים אצורים, עיון במאגרי נתונים אמיתיים או העלאת זוג קבצים משלך.",
    "Demo Mode stays the default, Manual Upload stays separate, and Dataset Browser / Pair Builder adds a third path that turns real catalog-backed assets into a verify-ready pair without touching File Explorer.": "מצב הדגמה נשאר ברירת המחדל, העלאה ידנית נשארת נפרדת, ודפדפן הנתונים / בונה הזוגות מוסיף מסלול שלישי שממיר נכסים אמיתיים מהקטלוג לזוג מוכן לאימות בלי לפתוח את מנהל הקבצים.",
    "Demo Mode": "מצב הדגמה",
    "Curated one-click verify": "אימות אצור בלחיצה אחת",
    "Dataset Browser": "דפדפן נתונים",
    "Dataset Picker": "בחירת מאגר נתונים",
    "Choose a browser-ready dataset from /api/catalog/datasets. Datasets without browser assets stay visible but disabled.": "בחר מאגר נתונים שמוכן לדפדוף מתוך /api/catalog/datasets. מאגרים בלי נכסי דפדוף נשארים גלויים אך אינם זמינים.",
    "Browse paginated real items, preview thumbnails, and assign them to side A or B.": "דפדוף בפריטים אמיתיים בעמודים, תצוגה מקדימה של תמונות ממוזערות ושיוך לצד A או B.",
    "Browser Filters": "מסנני דפדפן",
    "Narrow by metadata from the catalog browser endpoint. Active filters:": "צמצום לפי מטא־דאטה מנקודת הקצה של דפדפן הקטלוג. מסננים פעילים:",
    "Reset filters": "איפוס מסננים",
    "Shown when split metadata exists for the dataset.": "מוצג כאשר מטא־דאטה של פיצול קיימת במאגר.",
    "All splits": "כל הפיצולים",
    "Capture metadata stays dataset-aware and optional.": "מטא־דאטה של לכידה נשארת מודעת־מאגר ואופציונלית.",
    "All captures": "כל הלכידות",
    "Modality": "מודאליות",
    "Only values exposed by the current dataset appear here.": "כאן מופיעים רק ערכים שנחשפים על ידי המאגר הנוכחי.",
    "All modalities": "כל המודאליות",
    "All items": "כל הפריטים",
    "Eligible only": "זמינים בלבד",
    "Ineligible only": "לא זמינים בלבד",
    "Finger": "אצבע",
    "Use the catalog finger identifier when available.": "שימוש במזהה האצבע מהקטלוג כשהוא זמין.",
    "Sort": "מיון",
    "Uses the backend-supported browser sort values only.": "משתמש רק בערכי המיון לדפדפן שנתמכים בצד השרת.",
    "Default order": "סדר ברירת מחדל",
    "Split / subject / asset": "פיצול / נבדק / נכס",
    "Browser Items": "פריטי דפדפן",
    "Validation": "אימות תקינות",
    "Unavailable": "לא זמין",
    "Previous page": "עמוד קודם",
    "Next page": "עמוד הבא",
    "Pagination stays server-backed through": "הדפדוף נשאר מגובה שרת דרך",
    "Selection hint": "רמז בחירה",
    "Loading browser items": "טוען פריטי דפדפן",
    "Fetching the current dataset page from /api/catalog/dataset-browser.": "מביא את עמוד המאגר הנוכחי מתוך /api/catalog/dataset-browser.",
    "No items match the current filters": "אין פריטים שתואמים למסננים הנוכחיים",
    "Reset one or more filters to see more dataset-backed items.": "אפס מסנן אחד או יותר כדי לראות עוד פריטים מגובי מאגר.",
    "Refreshing the current page from the catalog browser...": "מרענן את העמוד הנוכחי מדפדפן הקטלוג...",
    "This item is still selectable even if the thumbnail could not be loaded.": "הפריט עדיין ניתן לבחירה גם אם התמונה הממוזערת לא נטענה.",
    "Preview unavailable": "תצוגה מקדימה לא זמינה",
    "Pair Status": "מצב הזוג",
    "Build a pair in two steps: choose side A, choose side B, then send both server-backed previews into Verify.": "בניית זוג בשני שלבים: בחירת צד A, בחירת צד B, ואז שליחת שתי התצוגות המקדימות מגובות השרת לאימות.",
    "Current dataset:": "מאגר נוכחי:",
    "Select a browser-ready dataset": "בחר מאגר נתונים שמוכן לדפדוף",
    "Failed to load the selected pair into Verify": "טעינת הזוג הנבחר לאימות נכשלה",
    "Verify inputs are synced with the selected pair": "קלטי האימות מסונכרנים עם הזוג הנבחר",
    "The probe and reference files now come from the dataset browser. Run Verification stays a separate action.": "קובצי הבדיקה והייחוס מגיעים עכשיו מדפדפן הנתונים. הרצת האימות נשארת פעולה נפרדת.",
    "One more step before running Verify": "עוד שלב אחד לפני הרצת האימות",
    "Use the selected pair as the verify pair to populate the probe/reference files from the server.": "השתמש בזוג הנבחר כזוג האימות כדי למלא מהשרת את קובצי הבדיקה והייחוס.",
    "Waiting for selection": "ממתין לבחירה",
    "Side": "צד",
    "Pick a dataset item to assign it to side": "בחר פריט מאגר כדי לשייך אותו לצד",
    "The pair stays usable even if the larger preview image is unavailable.": "הזוג נשאר שמיש גם אם תמונת התצוגה המקדימה הגדולה אינה זמינה.",
    "Clear": "ניקוי",
    "Swap A / B": "החלפת A / B",
    "Loading pair...": "טוען זוג...",
    "Pair loaded into Verify": "הזוג נטען לאימות",
    "Use as verify pair": "שימוש כזוג אימות",
    "Build a pair from real data": "בניית זוג מנתונים אמיתיים",
    "Manual Upload": "העלאה ידנית",
    "Bring your own two files": "בחירת שני קבצים משלך",
    "Default Path": "מסלול ברירת מחדל",
    "Curated first-run experience": "חוויית התחלה אצורה",
    "Browser-Ready Datasets": "מאגרי נתונים מוכנים לדפדוף",
    "Loaded from catalog datasets": "נטען מקטלוג מאגרי הנתונים",
    "Latest Context": "הקשר אחרון",
    "Active method": "שיטה פעילה",
    "All cases": "כל המקרים",
    "Choose a dataset": "בחר מאגר נתונים",
    "Showing": "מציג",
    "Offset": "היסט",
    "Active filters": "מסננים פעילים",
    "Generated": "נוצרו",
    "Failed to load dataset items": "טעינת פריטי המאגר נכשלה",
    "Fastest method": "השיטה המהירה ביותר",
    "UI Eligible": "זמין לממשק",
    "Maps directly to the ui_eligible query parameter.": "ממופה ישירות לפרמטר השאילתה ui_eligible.",
    "Subject ID": "מזהה נבדק",
    "Subject": "נבדק",
    "Free text because not every dataset exposes a fixed list.": "טקסט חופשי, כי לא כל מאגר חושף רשימה קבועה.",
    "e.g. 100001": "למשל 100001",
    "e.g. 1": "למשל 1",
    "Page Size": "גודל עמוד",
    "Keeps the browser paginated instead of loading whole datasets.": "שומר על דפדוף בעמודים במקום לטעון מאגרים שלמים.",
    "No run yet": "עדיין אין הרצה",
    "Waiting for first execution": "ממתין להרצה ראשונה",
    "Backend initialization issue detected": "זוהתה בעיית אתחול בשרת",
    "Curated demo asset is unavailable": "נכס הדמו האצור אינו זמין",
    "Clear saved Verify workspace": "ניקוי סביבת האימות השמורה",
    "Start with a ready-made verify case. Metadata comes from the catalog layer and the files are pulled from the server when you run.": "התחלה ממקרה אימות מוכן. המטא־דאטה מגיע משכבת הקטלוג והקבצים נמשכים מהשרת בזמן ההרצה.",
    "Selected Case": "מקרה נבחר",
    "Pin case": "הצמדת מקרה",
    "Unpin case": "ביטול הצמדה",
    "Run Selected Case": "הרצת המקרה הנבחר",
    "Pinned demo cases": "מקרי דמו מוצמדים",
    "Keep a small verify playlist ready across reloads without restoring old results.": "שמירת רשימת אימות קצרה בין טעינות מחדש בלי לשחזר תוצאות ישנות.",
    "Selected Browser Pair": "זוג דפדפן נבחר",
    "Selected Pair": "זוג נבחר",
    "Pair Builder": "בונה זוגות",
    "Method Settings": "הגדרות שיטה",
    "Method": "שיטה",
    "Capture A": "לכידה A",
    "Capture B": "לכידה B",
    "Threshold Mode": "מצב סף",
    "Threshold Value": "ערך סף",
    "Use method default": "שימוש בברירת המחדל של השיטה",
    "Custom threshold": "סף מותאם",
    "Ignored when threshold mode is set to default.": "לא בשימוש כאשר מצב הסף מוגדר לברירת מחדל.",
    "Execution Toggles": "אפשרויות הרצה",
    "Return overlay": "החזרת שכבת תצוגה",
    "Warm up matcher": "חימום מנוע ההתאמה",
    "Show outliers on canvas": "הצגת חריגים על הקנבס",
    "Show tentative on canvas": "הצגת התאמות זמניות על הקנבס",
    "Run Verification": "הרצת אימות",
    "Running verify...": "מריץ אימות...",
    "Running demo...": "מריץ דמו...",
    "Loading demo...": "טוען דמו...",
    "Warming matcher...": "מחמם מנוע התאמה...",
    "Latest Result": "תוצאה אחרונה",
    "The decision stays attached to the exact case or file pair that produced it.": "ההחלטה נשארת מחוברת למקרה או לזוג הקבצים המדויק שיצר אותה.",
    "Demo Result Context": "הקשר תוצאת דמו",
    "Browser Result Context": "הקשר תוצאת דפדפן",
    "Manual Result Context": "הקשר תוצאה ידנית",
    "Dataset / Split": "מאגר נתונים / פיצול",
    "Method Behavior": "התנהגות שיטה",
    "Using recommended method": "שימוש בשיטה המומלצת",
    "Asset Pair": "זוג נכסים",
    "Probe File": "קובץ בדיקה",
    "Reference File": "קובץ ייחוס",
    "Verification request in progress": "בקשת אימות מתבצעת",
    "Verification failed": "האימות נכשל",
    "Try again": "נסה שוב",
    "Raw metrics": "מדדים גולמיים",
    "The storytelling layer stays on top, while the original metric block remains available underneath.": "שכבת הסיפור נשארת למעלה, ובלוק המדדים המקורי נשאר זמין מתחת.",
    "Hide raw metrics": "הסתרת מדדים גולמיים",
    "Show raw metrics": "הצגת מדדים גולמיים",
    "No drawable overlay available": "אין שכבת תצוגה שניתן לצייר",
    "No result yet": "עדיין אין תוצאה",
    "Server-backed execution": "הרצה מגובה שרת",
    "Upload fingerprint": "העלאת טביעת אצבע",
    "Drag & drop or click to select an image": "גרירה ושחרור או לחיצה לבחירת תמונה",
    "Remove file": "הסרת קובץ",
    "Ready": "מוכן",
    "Match Visualization": "תצוגת התאמות",
    "Connecting corresponding minutiae/patches": "חיבור נקודות ותבניות תואמות",
    "Inlier": "תואם פנימי",
    "Outlier": "חריג",
    "Decision": "החלטה",
    "Similarity score": "ציון דמיון",
    "MATCH CONFIRMED": "התאמה אושרה",
    "NO MATCH": "אין התאמה",
    "Threshold": "סף",
    "Latency": "זמן תגובה",
    "Overlay": "שכבת תצוגה",
    "Raw matches": "התאמות גולמיות",
    "Inliers": "תואמים פנימיים",
    "Keypoints A/B": "נקודות מפתח A/B",
    "Backbone": "מודל בסיס",
    "Embed A/B": "Embedding A/B",
    "Tentative / Inliers": "זמניות / פנימיות",
    "Mean inlier sim": "דמיון פנימי ממוצע",
    "Latency breakdown": "פירוט זמני תגובה",

    "Identification Workspace": "סביבת זיהוי",
    "Demo, browser, and operational 1:N search in one Identification workspace.": "חיפוש 1:N בדמו, בדפדפן ובמצב תפעולי בתוך סביבת זיהוי אחת.",
    "Demo Mode stays curated, Browser Mode adds identity-aware gallery selection plus a dataset-backed probe browser, and Operational Mode keeps the manual enroll/search/delete controls intact.": "מצב הדגמה נשאר אצור, מצב דפדפן מוסיף בחירת גלריה מודעת־זהות ודפדפן בדיקות מגובה מאגר נתונים, ומצב תפעולי שומר על בקרות רישום/חיפוש/מחיקה ידניות.",
    "Browser Mode": "מצב דפדפן",
    "Operational Mode": "מצב תפעולי",
    "Seed a curated gallery, pick a probe, and run a guided 1:N flow.": "הזרעת גלריה אצורה, בחירת בדיקה והרצת תהליך 1:N מונחה.",
    "Keep full control over stats, enroll, manual search, and delete workflows.": "שמירה על שליטה מלאה בסטטיסטיקות, רישום, חיפוש ידני ותהליכי מחיקה.",
    "Recommended first path": "מסלול ראשון מומלץ",
    "Browser workspace": "סביבת דפדפן",
    "Pick one dataset, choose the gallery identities that should be enrolled into an isolated browser store, then choose a single probe asset from the dataset browser.": "בחר מאגר אחד, בחר את זהויות הגלריה שיירשמו למאגר דפדפן מבודד, ואז בחר נכס בדיקה יחיד מדפדפן הנתונים.",
    "Choose this dataset to browse probe assets and build a catalog-backed 1:N search context for Identification.": "בחר את המאגר הזה כדי לדפדף בנכסי בדיקה ולבנות הקשר חיפוש 1:N מגובה קטלוג עבור זיהוי.",
    "Browser mode requires both browser assets and identify-gallery metadata for this dataset.": "מצב דפדפן דורש גם נכסי דפדפוף וגם מטא־דאטה של גלריית זיהוי עבור המאגר הזה.",
    "The catalog does not currently expose any datasets with both identify-gallery metadata and browser assets.": "הקטלוג אינו חושף כרגע מאגרים שיש בהם גם מטא־דאטה של גלריית זיהוי וגם נכסי דפדוף.",
    "Gallery selection": "בחירת גלריה",
    "Gallery cards stay identity-aware and come directly from the identify-gallery catalog semantics.": "כרטיסי הגלריה נשארים מודעי־זהות ומגיעים ישירות מסמנטיקת קטלוג גלריית הזיהוי.",
    "Probe browser": "דפדפן בדיקות",
    "Reuse the dataset browser infrastructure from Verify without any A/B pair-builder semantics.": "שימוש חוזר בתשתית דפדפן הנתונים של אימות, בלי סמנטיקת בניית זוג A/B.",
    "Run browser identification": "הרצת זיהוי דפדפן",
    "The browser flow seeds the selected gallery into its isolated store and then runs the official identification endpoint against that seeded context.": "תהליך הדפדפן מזריע את הגלריה הנבחרת למאגר המבודד שלה, ואז מריץ את נקודת הקצה הרשמית לזיהוי מול ההקשר שהוזרע.",
    "Choose a dataset, select gallery identities and a probe, then run the browser workflow to populate this panel.": "בחר מאגר נתונים, זהויות גלריה ובדיקה, ואז הרץ את תהליך הדפדפן כדי למלא את הלוח הזה.",
    "Choose a dataset, seed selected identities into an isolated browser gallery, and run 1:N with a browser-picked probe.": "בחר מאגר נתונים, הזרע זהויות נבחרות לגלריית דפדפן מבודדת, והרץ 1:N עם בדיקה שנבחרה בדפדפן.",
    "Demo identities": "זהויות דמו",
    "Server-backed gallery cards": "כרטיסי גלריה מגובי שרת",
    "Browser datasets": "מאגרי דפדפן",
    "Catalog-ready for guided 1:N": "מוכן מהקטלוג ל־1:N מונחה",
    "Demo probes": "בדיקות דמו",
    "Curated 1:N stories": "סיפורי 1:N אצורים",
    "Browser store status": "מצב מאגר הדפדפן",
    "Seeded": "מוזרע",
    "Not seeded": "לא הוזרע",
    "Empty": "ריק",
    "Shortlist returned zero candidates": "הרשימה הקצרה החזירה אפס מועמדים",
    "Clear saved Identification workspace": "ניקוי סביבת הזיהוי השמורה",
    "Operational controls preserved": "הבקרות התפעוליות נשמרו",
    "Guided paths": "מסלולים מונחים",
    "Official endpoint": "נקודת קצה רשמית",
    "Isolated seeding": "הזרעה מבודדת",
    "Candidate shortlist": "רשימת מועמדים קצרה",
    "Top candidates surfaced from the official 1:N response, including retrieval and re-rank scores.": "המועמדים המובילים מתוך תגובת 1:N הרשמית, כולל ציוני שליפה ודירוג מחדש.",
    "Accepted": "התקבל",
    "Not accepted": "לא התקבל",
    "Rank": "דירוג",
    "Person": "אדם",
    "Masked ID": "מזהה מוסתר",
    "Random ID": "מזהה אקראי",
    "National ID": "מזהה לאומי",
    "Capture": "לכידה",
    "Enroll": "רישום",
    "Retrieval": "שליפה",
    "Re-rank": "דירוג מחדש",
    "Candidate ranking": "דירוג מועמדים",
    "Demo identity gallery": "גלריית זהויות דמו",
    "These are the curated identities that the demo seeding flow will enroll.": "אלו הזהויות האצורות שתהליך הזרעת הדמו ירשום.",
    "Gallery identities": "זהויות גלריה",
    "Select one or more catalog identities to seed the browser-isolated 1:N gallery.": "בחר זהות קטלוג אחת או יותר כדי להזריע את גלריית 1:N המבודדת של הדפדפן.",
    "This dataset does not currently expose any identify-gallery identities for Browser mode.": "המאגר הזה אינו חושף כרגע זהויות גלריית זיהוי עבור מצב דפדפן.",
    "In gallery": "בגלריה",
    "Selected probe": "בדיקה נבחרת",
    "Choose one asset from the dataset browser without opening the system file picker.": "בחר נכס אחד מדפדפן הנתונים בלי לפתוח את בוחר הקבצים של המערכת.",
    "Clear probe": "ניקוי בדיקה",
    "No probe selected yet. Pick an item below to use it as the single browser probe.": "עדיין לא נבחרה בדיקה. בחר פריט למטה כדי להשתמש בו כבדיקת הדפדפן היחידה.",
    "Loading dataset browser": "טוען דפדפן נתונים",
    "Fetching the current page of server-backed browser assets.": "מביא את העמוד הנוכחי של נכסי הדפדפן מגובי השרת.",
    "Failed to load browser assets": "טעינת נכסי הדפדפן נכשלה",
    "No browser assets match the current filters": "אין נכסי דפדפן שתואמים למסננים הנוכחיים",
    "Reset one or more filters to see more probe candidates.": "אפס מסנן אחד או יותר כדי לראות מועמדי בדיקה נוספים.",
    "Probe": "בדיקה",
    "Use as probe": "שימוש כבדיקה",
    "Replace probe": "החלפת בדיקה",
    "Latest browser result": "תוצאת דפדפן אחרונה",
    "No browser run yet": "עדיין אין הרצת דפדפן",
    "Running browser identification": "מריץ זיהוי דפדפן",
    "Seeding the browser-selected gallery and waiting for the 1:N response.": "מזריע את הגלריה שנבחרה בדפדפן וממתין לתגובת 1:N.",
    "Browser identification failed": "זיהוי הדפדפן נכשל",
    "No browser identification result yet": "עדיין אין תוצאת זיהוי דפדפן",
    "The result panel will show the seeded-gallery outcome, shortlist, methods, and latency after the first browser run.": "לוח התוצאות יציג את תוצאת הגלריה שהוזרעה, הרשימה הקצרה, השיטות וזמן התגובה אחרי הרצת הדפדפן הראשונה.",
    "Dataset": "מאגר נתונים",
    "Probe asset": "נכס בדיקה",
    "Methods": "שיטות",
    "Shortlist": "רשימה קצרה",
    "Running...": "מריץ...",
    "Selection summary": "סיכום בחירה",
    "Search controls": "בקרות חיפוש",
    "Tune the browser-backed search context before seeding the gallery and running the official 1:N endpoint.": "כוון את הקשר החיפוש מגובה הדפדפן לפני הזרעת הגלריה והרצת נקודת הקצה הרשמית של 1:N.",
    "Retrieval method": "שיטת שליפה",
    "Re-rank method": "שיטת דירוג מחדש",
    "Shortlist size": "גודל רשימה קצרה",
    "Show advanced filters": "הצגת מסננים מתקדמים",
    "Hide advanced filters": "הסתרת מסננים מתקדמים",
    "Seed gallery and run": "הזרעת גלריה והרצה",
    "Reset browser store": "איפוס מאגר הדפדפן",
    "Name pattern": "תבנית שם",
    "National ID pattern": "תבנית מזהה לאומי",
    "Created from": "נוצר מתאריך",
    "Created to": "נוצר עד תאריך",
    "Browser Mode still reaches": "מצב דפדפן עדיין מגיע אל",
    ", but it first seeds the selected catalog identities into the isolated browser store so the 1:N run uses a real seeded gallery instead of UI-only state.": ", אבל קודם הוא מזריע את זהויות הקטלוג הנבחרות למאגר הדפדפן המבודד, כך שהרצת 1:N משתמשת בגלריה מוזרעת אמיתית ולא במצב UI בלבד.",
    "Leave empty to use the backend default.": "השאר ריק כדי להשתמש בברירת המחדל של השרת.",
    "browser-seeded identities tracked": "זהויות דפדפן מוזרעות במעקב",
    "One of the identification endpoints appears to have failed during startup or lazy initialization. Keep the original error visible and treat this as a release-readiness blocker before retrying the flow.": "נראה שאחת מנקודות הקצה של הזיהוי נכשלה בזמן עלייה או אתחול עצל. השאר את השגיאה המקורית גלויה והתייחס לכך כחסם מוכנות לשחרור לפני ניסיון חוזר.",
    "The demo request succeeded, but the backend returned an empty shortlist. This is a valid negative path and the UI keeps it readable.": "בקשת הדמו הצליחה, אבל השרת החזיר רשימה קצרה ריקה. זהו מסלול שלילי תקין והממשק שומר עליו קריא.",
    "Demo and Browser add guided product workflows above the existing capabilities. Stats, enroll, manual search, and delete remain available here unchanged.": "דמו ודפדפן מוסיפים תהליכי מוצר מונחים מעל היכולות הקיימות. סטטיסטיקות, רישום, חיפוש ידני ומחיקה נשארים זמינים כאן ללא שינוי.",
    "Demo gives a curated walkthrough first, while Browser lets you build a catalog-backed search context without falling back to file uploads.": "דמו נותן תחילה תהליך אצור, בעוד שדפדפן מאפשר לבנות הקשר חיפוש מגובה קטלוג בלי לחזור להעלאת קבצים.",
    "The guided flow still reaches": "התהליך המונחה עדיין מגיע אל",
    "; there is no parallel identification engine.": "; אין מנוע זיהוי מקביל.",
    "Browser-selected galleries seed into their own resettable store so operational enrollments stay untouched.": "גלריות שנבחרו בדפדפן מוזרעות למאגר ניתן־לאיפוס משלהן, כך שרישומים תפעוליים לא משתנים.",
    "Pinned probes": "בדיקות מוצמדות",
    "Keep your key demo entries one click away across reloads.": "שמירת פריטי הדמו החשובים במרחק לחיצה גם אחרי טעינה מחדש.",
    "Recent probes": "בדיקות אחרונות",
    "Recently selected or executed probes stay available as lightweight continuity only.": "בדיקות שנבחרו או הורצו לאחרונה נשארות זמינות כרציפות קלה בלבד.",
    "Expected vs actual": "צפוי מול בפועל",
    "Expected": "צפוי",
    "match": "התאמה",
    "no match": "אין התאמה",
    "Match": "התאמה",
    "No match": "אין התאמה",
    "MATCH": "התאמה",
    "Actual": "בפועל",
    "Latency details will appear once the backend returns a completed result.": "פירוט זמני התגובה יופיע לאחר שהשרת יחזיר תוצאה מלאה.",
    "Start in Demo Mode, switch to Browser Mode for guided catalog-backed 1:N, then drop into Operational Mode for manual controls.": "התחלה במצב הדגמה, מעבר למצב דפדפן עבור 1:N מונחה מגובה קטלוג, ואז מעבר למצב תפעולי לבקרות ידניות.",
    "Probe context": "הקשר בדיקה",
    "Confidence band": "רצועת ביטחון",
    "Latest demo result": "תוצאת דמו אחרונה",
    "Negative path": "מסלול שלילי",
    "Probe cases": "מקרי בדיקה",
    "Choose a ready-made probe without touching File Explorer or manual uploads.": "בחירת בדיקה מוכנה בלי לפתוח את מנהל הקבצים או העלאה ידנית.",
    "Top candidate": "מועמד מוביל",
    "Not active": "לא פעיל",
    "Not selected": "לא נבחר",
    "Retrieval score": "ציון שליפה",
    "Re-rank score": "ציון דירוג מחדש",
    "Plain": "רגילה",
    "Roll": "גלגול",
    "Contactless": "ללא מגע",
    "Contact-based": "מבוססת מגע",
    "Vector methods": "שיטות וקטוריות",

    "Benchmarks": "מדדי ביצועים",
    "Curated full results": "תוצאות מלאות אצורות",
    "Partial evidence": "ראיות חלקיות",
    "Validated showcase": "תצוגה מאומתת",
    "Curated full benchmark results for validated showcase runs. Compare accuracy, EER, latency, and method evidence without browsing historical tiers.": "תוצאות מדדי ביצועים מלאות ואצורות להרצות תצוגה מאומתות. השוואת דיוק, EER, זמן תגובה וראיות שיטה בלי לעיין בשכבות היסטוריות.",
    "Split": "פיצול",
    "Sort by": "מיון לפי",
    "Refresh results": "רענון תוצאות",
    "Current scope": "תחום נוכחי",
    "Loading dataset": "טוען מאגר נתונים",
    "Run family": "משפחת הרצה",
    "Resolving split": "מאתר פיצול",
    "Showing curated full benchmark results.": "מציג תוצאות מדדי ביצועים מלאות ואצורות.",
    "Executive summary": "תקציר מנהלים",
    "Showcase winners and trade-offs": "מובילי התצוגה ופשרות",
    "Full comparison": "השוואה מלאה",
    "Method comparison table": "טבלת השוואת שיטות",
    "Compare the validated showcase methods on the active dataset and split. Missing values remain stable as N/A so the comparison stays readable.": "השוואת שיטות התצוגה המאומתות במאגר ובפיצול הפעילים. ערכים חסרים נשארים כ־N/A כדי שההשוואה תישאר קריאה.",
    "Loading curated comparison rows": "טוען שורות השוואה אצורות",
    "Reading validated full benchmark rows for the active dataset and split.": "קורא שורות מדדי ביצועים מלאות ומאומתות עבור המאגר והפיצול הפעילים.",
    "Failed to load benchmark summary": "טעינת סיכום המדדים נכשלה",
    "Failed to load comparison rows": "טעינת שורות ההשוואה נכשלה",
    "Retry": "נסה שוב",
    "No showcase winners for this selection": "אין מובילי תצוגה עבור הבחירה הזו",
    "Choose another dataset or split with curated full benchmark results.": "בחר מאגר נתונים או פיצול אחר עם תוצאות מדדים מלאות ואצורות.",
    "No curated full benchmark results for this selection": "אין תוצאות מדדים מלאות ואצורות עבור הבחירה הזו",
    "Choose another dataset or split to continue browsing the showcase.": "בחר מאגר נתונים או פיצול אחר כדי להמשיך לעיין בתצוגה.",
    "Evidence panel": "לוח ראיות",
    "Select a comparison row to inspect run provenance, artifacts, and validation state.": "בחר שורת השוואה כדי לבדוק מקור הרצה, ארטיפקטים ומצב אימות.",
    "Selected run": "הרצה נבחרת",
    "Artifacts": "ארטיפקטים",
    "Provenance": "מקור",
    "Best accuracy": "הדיוק הטוב ביותר",
    "Lowest EER": "EER הנמוך ביותר",
    "Lowest latency": "זמן התגובה הנמוך ביותר",
    "best accuracy": "הדיוק הטוב ביותר",
    "lowest EER": "EER הנמוך ביותר",
    "lowest latency": "זמן התגובה הנמוך ביותר",
};

const HEBREW_PATTERNS: Array<[RegExp, (...matches: string[]) => string]> = [
    [/^(\d+) comparison rows$/, (count) => `${count} שורות השוואה`],
    [/^(\d+) methods$/, (count) => `${count} שיטות`],
    [/^(\d+) selected$/, (count) => `${count} נבחרו`],
    [/^Showing (.+)$/, (summary) => `מציג ${translateHebrew(summary)}`],
    [/^(.+) of (.+) items$/, (count, total) => `${count} מתוך ${total} פריטים`],
    [/^Offset (.+)$/, (offset) => `היסט ${offset}`],
    [/^Active filters (.+)$/, (count) => `מסננים פעילים ${count}`],
    [/^Generated (.+)$/, (count) => `נוצרו ${count}`],
    [/^(.+) items$/, (count) => `${count} פריטים`],
    [/^Sorted by (.+)$/, (label) => `מיון לפי ${translateHebrew(label)}`],
    [/^Rank (\d+)$/, (rank) => `דירוג ${rank}`],
    [/^Subject (.+)$/, (subject) => `נבדק ${subject}`],
    [/^Capture: (.+)$/, (capture) => `לכידה: ${translateHebrew(capture)}`],
    [/^Finger: (.+)$/, (finger) => `אצבע: ${translateHebrew(finger)}`],
    [/^Subject: (.+)$/, (subject) => `נבדק: ${subject}`],
    [/^Modality: (.+)$/, (modality) => `מודאליות: ${translateHebrew(modality)}`],
    [/^Split: (.+)$/, (split) => `פיצול: ${translateHebrew(split)}`],
    [/^Side (.+)$/, (side) => `צד ${side}`],
    [/^Pick a dataset item to assign it to side (.+)\.$/, (side) => `בחר פריט מאגר כדי לשייך אותו לצד ${side}.`],
    [/^Replace (.+)$/, (side) => `החלפת ${side}`],
    [/^Cancel Replace (.+)$/, (side) => `ביטול החלפת ${side}`],
    [/^Clear (.+)$/, (side) => `ניקוי ${side}`],
    [/^Selected as (.+)$/, (side) => `נבחר כ־${side}`],
    [/^(Choose|Replace) (A|B)$/, (action, side) => `${action === "Replace" ? "החלפת" : "בחירת"} ${side}`],
    [/^Choose (A|B) or (A|B) first$/, (first, second) => `בחר קודם ${first} או ${second}`],
    [/^Enroll (.+)$/, (capture) => `רישום ${translateHebrew(capture)}`],
    [/^Probe (.+)$/, (capture) => `בדיקה ${translateHebrew(capture)}`],
    [/^Shortlist (.+)$/, (size) => `רשימה קצרה ${size}`],
    [/^Method (.+)$/, (method) => `שיטה ${method}`],
    [/^Recommended (.+)$/, (method) => `מומלץ ${method}`],
    [/^Threshold (.+)$/, (value) => `סף ${translateHebrew(value)}`],
    [/^Max (\d+) canvas matches$/, (count) => `עד ${count} התאמות קנבס`],
    [/^(\d+) pinned$/, (count) => `${count} מוצמדים`],
    [/^(\d+) browser-seeded identities tracked$/, (count) => `${count} זהויות דפדפן מוזרעות במעקב`],
    [/^Override from (.+)$/, (method) => `החלפה מ־${method}`],
    [/^(.+) matches$/, (count) => `${count} התאמות`],
];

const SKIPPED_TAGS = new Set(["CODE", "KBD", "PRE", "SAMP", "SCRIPT", "STYLE", "SVG"]);
const TRANSLATED_ATTRIBUTES = ["aria-label", "alt", "placeholder", "title"] as const;

const originalText = new WeakMap<Text, string>();
const originalAttributes = new WeakMap<Element, Map<string, string>>();
let hasLocalizedContent = false;

function normalizeText(value: string): string {
    return value.replace(/\s+/g, " ").trim();
}

function withOriginalWhitespace(original: string, translated: string): string {
    const leading = original.match(/^\s*/)?.[0] ?? "";
    const trailing = original.match(/\s*$/)?.[0] ?? "";
    return `${leading}${translated}${trailing}`;
}

function translateHebrew(value: string): string {
    const normalized = normalizeText(value);
    if (!normalized) {
        return value;
    }

    const exact = HEBREW_TEXT[normalized];
    if (exact) {
        return withOriginalWhitespace(value, exact);
    }

    for (const [pattern, translate] of HEBREW_PATTERNS) {
        const match = normalized.match(pattern);
        if (match) {
            return withOriginalWhitespace(value, translate(...match.slice(1)));
        }
    }

    return value;
}

function shouldSkipTextNode(node: Text): boolean {
    let parent = node.parentElement;
    while (parent) {
        if (SKIPPED_TAGS.has(parent.tagName) || parent.hasAttribute("data-no-localize")) {
            return true;
        }
        parent = parent.parentElement;
    }

    return false;
}

function localizeTextNode(node: Text, language: AppLanguage): void {
    if (shouldSkipTextNode(node)) {
        return;
    }

    const original = originalText.get(node) ?? node.nodeValue ?? "";
    if (language === "en") {
        if (originalText.has(node) && node.nodeValue !== original) {
            node.nodeValue = original;
        }
        return;
    }

    const translated = translateHebrew(original);
    if (translated !== original) {
        originalText.set(node, original);
        hasLocalizedContent = true;
        if (node.nodeValue !== translated) {
            node.nodeValue = translated;
        }
    }
}

function localizeElementAttributes(element: Element, language: AppLanguage): void {
    for (const attribute of TRANSLATED_ATTRIBUTES) {
        const currentValue = element.getAttribute(attribute);
        if (!currentValue) {
            continue;
        }

        const attributeMap = originalAttributes.get(element);
        const original = attributeMap?.get(attribute) ?? currentValue;

        if (language === "en") {
            if (attributeMap?.has(attribute) && currentValue !== original) {
                element.setAttribute(attribute, original);
            }
            continue;
        }

        const translated = translateHebrew(original);
        if (translated !== original) {
            const nextAttributeMap = attributeMap ?? new Map<string, string>();
            nextAttributeMap.set(attribute, original);
            originalAttributes.set(element, nextAttributeMap);
            hasLocalizedContent = true;
            if (currentValue !== translated) {
                element.setAttribute(attribute, translated);
            }
        }
    }
}

function localizeTree(root: Element, language: AppLanguage): void {
    const textWalker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let textNode = textWalker.nextNode();
    while (textNode) {
        localizeTextNode(textNode as Text, language);
        textNode = textWalker.nextNode();
    }

    localizeElementAttributes(root, language);
    root.querySelectorAll("*").forEach((element) => localizeElementAttributes(element, language));
}

export function useFeatureDomLocalization(language: AppLanguage, refreshKey?: string): void {
    useEffect(() => {
        const root = document.querySelector(".app-content");
        if (!root) {
            return undefined;
        }

        if (language === "en" && !hasLocalizedContent) {
            return undefined;
        }

        localizeTree(root, language);

        if (language === "en") {
            return undefined;
        }

        const frameId = window.requestAnimationFrame(() => {
            localizeTree(root, language);
        });
        const timeoutId = window.setTimeout(() => {
            localizeTree(root, language);
        }, 250);

        return () => {
            window.cancelAnimationFrame(frameId);
            window.clearTimeout(timeoutId);
        };
    }, [language, refreshKey]);
}
