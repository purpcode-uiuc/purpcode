# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt

# latex required
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)

# Data provided
data = {
    "AWS credentials logged": 50,
    "AWS insecure transmission CDK": 50,
    "AWS missing encryption CDK": 50,
    "AWS missing encryption of sensitive data cdk": 50,
    "Clear text credentials": 50,
    "Cross-site request forgery": 56,
    "Cross-site scripting": 147,
    "Deserialization of untrusted object": 50,
    "Empty Password": 17,
    "Garbage collection prevention in multiprocessing": 58,
    "Hardcoded IP address": 50,
    "Hardcoded credentials": 144,
    "Improper authentication": 70,
    "Improper certificate validation": 44,
    "Improper input validation": 75,
    "Improper privilege management": 8,
    "Improper resource exposure": 70,
    "Improper sanitization of wildcards or matching symbols": 52,
    "Insecure CORS policy": 58,
    "Insecure Socket Bind": 66,
    "Insecure connection using unencrypted protocol": 83,
    "Insecure cookie": 64,
    "Insecure cryptography": 130,
    "Insecure hashing": 282,
    "Insecure temporary file or directory": 125,
    "Integer overflow": 50,
    "LDAP injection": 54,
    "Log injection": 82,
    "Loose file permissions": 241,
    "Missing Authorization CDK": 50,
    "Mutually exclusive call": 50,
    "OS command injection": 1411,
    "Override of reserved variable names in a Lambda function": 55,
    "Path traversal": 223,
    "Public method parameter validation": 273,
    "Resource leak": 1516,
    "S3 partial encrypt CDK": 50,
    "SQL injection": 106,
    "Socket connection timeout": 109,
    "Spawning a process without main module": 52,
    "URL redirection to untrusted site": 70,
    "Unauthenticated Amazon SNS unsubscribe requests might succeed": 50,
    "Unauthenticated LDAP requests": 50,
    "Unrestricted upload of dangerous file type": 70,
    "Unsafe Cloudpickle Load": 51,
    "Unsanitized input is run as code": 351,
    "Untrusted AMI images": 50,
    "Usage of an API that is not recommended": 17,
    "Usage of an API that is not recommended - High Severity": 29,
    "Usage of an API that is not recommended - Medium Severity": 1390,
    "Using AutoAddPolicy or WarningPolicy": 4,
    "Weak algorithm used for Password Hashing": 108,
    "Weak obfuscation of web request": 52,
    "XML External Entity": 19,
    "XPath injection": 51,
    "Zip bomb attack": 56,
}


# Prepare data: Top N and 'Others'
sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
top_n_count = 10
top_n_labels_orig = list(sorted_data.keys())[:top_n_count]
top_n_freqs = list(sorted_data.values())[:top_n_count]
top_n_ratio = [(f / sum(sorted_data.values())) for f in top_n_freqs]
other_size = sum(list(sorted_data.values())[top_n_count:])

# Create legend labels with frequencies
max_label_length = 64
plot_labels_for_legend = []
for i in range(len(top_n_labels_orig)):
    label_text = top_n_labels_orig[i].split(" - ")[0]
    freq = top_n_freqs[i]
    ratio = top_n_ratio[i]
    if len(label_text) > max_label_length:
        truncated_label_text = label_text[: max_label_length - 3] + "..."
    else:
        truncated_label_text = label_text
    plot_labels_for_legend.append(f"{truncated_label_text} ({ratio * 100:.1f}\\%)")

# Determine plot_sizes for pie chart and add 'Others' label if needed
if other_size > 0:
    plot_labels_for_legend.append(f"Others")  # MODIFIED: Added frequency for Others
    plot_sizes = top_n_freqs + [other_size]
else:
    plot_sizes = top_n_freqs


grouped_labels = []
grouped_counts = []
others_count = 0

for k, v in sorted_data.items():
    if v >= 144:
        grouped_labels.append(k)
        grouped_counts.append(v)
    else:
        others_count += v

grouped_labels.append("Others")
grouped_counts.append(others_count)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))


def make_autopct(values):
    def my_autopct(pct):
        return f"{pct:.1f}%"

    return my_autopct


# Styling
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))

# Create a color map
num_colors = len(plot_sizes)
colors_palette = [
    "#f7c59f",  # Soft peach
    "#ffb58b",  # Warm coral
    "#ffd48a",  # Pastel amber
    "#fff0a5",  # Light butter‑yellow
    "#e9e3a4",  # Sandstone
    "#d8f0a1",  # Pale pistachio
    "#c1e8b0",  # Mint‑melon
    "#b8e8d4",  # Icy aqua
    "#cde0ff",  # Powder periwinkle
    "#d8c7ff",  # Lilac
    "lightgray",
]

final_colors = [colors_palette[i % len(colors_palette)] for i in range(num_colors)]

wedges, texts, autotexts = ax.pie(
    plot_sizes,  # This now correctly reflects top N + Others (if any)
    # autopct="%1.1f\\%%",
    startangle=140,
    pctdistance=0.75,
    colors=final_colors,
    wedgeprops=dict(width=0.5, edgecolor="w"),
    textprops={"fontsize": 16},
    explode=[0.05 if label == "Others" else 0.03 for label in grouped_labels],
    autopct=make_autopct(grouped_counts),
)

for val, txt in zip(plot_sizes, autotexts):
    pct = val / sum(plot_sizes) * 100
    if pct > 15:
        txt.set_fontsize(18)
        txt.set_text(r"\textbf{" + txt.get_text() + r"}")

plt.setp(autotexts, size=11, weight="bold", color="black")
ax.axis("equal")

plt.subplots_adjust(left=0.1, right=0.85)
legend = ax.legend(
    wedges,
    plot_labels_for_legend,  # This now includes frequencies
    title="\\textbf{Top CodeGuru Detections}",
    title_fontsize="12",
    loc="center left",
    bbox_to_anchor=(0.95, 0.5),
    fontsize=11,  # May need to adjust if labels with freq are too long
    frameon=False,
    shadow=False,
)


plt.savefig(
    "cwepie.png",  # New filename
    bbox_extra_artists=(legend,),
    bbox_inches="tight",
    dpi=300,
    pad_inches=-0.05,  # User's custom padding
)
plt.savefig(
    "cwepie.pdf",  # New filename
    bbox_extra_artists=(legend,),
    bbox_inches="tight",
    pad_inches=-0.05,  # User's custom padding
)
